"""Quality-gate newly generated Day 4 training examples, without modifying them.

Ensures valid messages/JSON, grounded slot values by current audit heuristics,
and no exact user prompt overlaps against frozen evaluation or replay data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_sft_v2_data import inspect_action, iter_records, normalize


def question(record):
    if isinstance(record.get("question"), str):
        return normalize(record["question"])
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return normalize(msg.get("content", ""))
    return None


def load_questions(path):
    return {q for _, record in iter_records(path) if (q := question(record))}


def check_record(record, line):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Line {line}: expected system/user/assistant messages")
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Line {line}: unexpected role order")
    if not all(isinstance(m.get("content"), str) for m in messages):
        raise ValueError(f"Line {line}: non-text message")
    user = messages[1]["content"]
    action = json.loads(messages[2]["content"])
    if not isinstance(action, dict):
        raise ValueError(f"Line {line}: action not a JSON object")
    kind = action.get("type")
    if kind == "state_update":
        if set(action) != {"type", "state"} or not isinstance(action["state"], dict):
            raise ValueError(f"Line {line}: invalid state structure")
        state = action["state"]
        allowed = {
            "origin", "duration_days", "traveller_group", "traveller_count",
            "budget_inr", "budget_basis", "interests",
        }
        if not set(state).issubset(allowed):
            raise ValueError(f"Line {line}: unexpected state field")
        if "destination" in state or "date_expression" in state:
            raise ValueError(f"Line {line}: invented destination or date")
        if not ("origin" in state and "duration_days" in state):
            raise ValueError(f"Line {line}: missing origin/duration")
        if ("budget_inr" in state) != ("budget_basis" in state):
            raise ValueError(f"Line {line}: budget and basis must appear together")
        if state.get("traveller_group") in ("friends", "family") and "traveller_count" in state:
            raise ValueError(f"Line {line}: unsupported group-size default")
        if state.get("traveller_group") == "solo" and state.get("traveller_count") != 1:
            raise ValueError(f"Line {line}: wrong solo count")
        if state.get("traveller_group") == "couple" and state.get("traveller_count") != 2:
            raise ValueError(f"Line {line}: wrong couple count")
        if state.get("traveller_group") == "family_parents" and state.get("traveller_count") != 3:
            raise ValueError(f"Line {line}: wrong parents count")
    elif kind == "tool_call":
        if set(action) != {"type", "tool", "arguments"}:
            raise ValueError(f"Line {line}: invalid tool structure")
        if action["tool"] not in ("weather", "destination_knowledge"):
            raise ValueError(f"Line {line}: unsupported Day 4 training tool")
        if not isinstance(action["arguments"], dict):
            raise ValueError(f"Line {line}: invalid tool arguments")
    elif kind == "message":
        if set(action) != {"type", "message"}:
            raise ValueError(f"Line {line}: invalid message structure")
        if "?" not in action["message"]:
            raise ValueError(f"Line {line}: missing clarification question")
    else:
        raise ValueError(f"Line {line}: unknown action type {kind!r}")

    warnings = inspect_action(action, normalize(user))
    if warnings:
        raise ValueError(f"Line {line}: audit warnings {warnings}; user={user!r}")
    return kind


def main():
    parser = argparse.ArgumentParser(description="Validate Day 4 generated SFT JSONL")
    parser.add_argument("--training-file", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, action="append", default=[])
    parser.add_argument("--existing-train", type=Path, action="append", default=[])
    parser.add_argument("--expected-count", type=int, default=1640)
    args = parser.parse_args()

    if not args.training_file.is_file():
        raise FileNotFoundError(args.training_file)
    disallow = set()
    for path in args.holdout + args.existing_train:
        if not path.is_file():
            raise FileNotFoundError(path)
        disallow.update(load_questions(path))

    counts = Counter()
    unique = set()
    overlap = []
    for line, record in iter_records(args.training_file):
        prompt = question(record)
        if not prompt or prompt in unique:
            raise ValueError(f"Missing/duplicate prompt at line {line}")
        unique.add(prompt)
        if prompt in disallow:
            overlap.append(line)
        counts[check_record(record, line)] += 1
    if overlap:
        raise ValueError(f"Exact eval/replay prompt overlap at lines: {overlap[:20]}")
    if len(unique) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} cases, got {len(unique)}")
    sha = hashlib.sha256(args.training_file.read_bytes()).hexdigest()
    print("DAY 4 GROUNDED TRAINING DATA — VALIDATED")
    print("Unique conversations:", len(unique))
    print("Action types:", dict(counts))
    print("Prior/held-out exact prompt overlaps:", 0)
    print("Heuristic label warnings:", 0)
    print("Training SHA256:", sha)
    print("File:", args.training_file)
    print("Note: heuristic pass cannot guarantee all synthetic labels are semantically correct.")


if __name__ == "__main__":
    main()
