"""Apply five manually reviewed conflicting-party clarifications to SFT v2 candidate.

Produces a NEW proposed final JSONL and a change manifest. Never edits input.
Scope is deliberately restricted to the five reviewed JSONL line numbers.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from audit_sft_v2_data import iter_records, normalize, inspect_action

REVIEWED = {
    172: {"origin": "Kochi", "phrase": "our full budget"},
    417: {"origin": "Chennai", "phrase": "all of us"},
    421: {"origin": "Kolkata", "phrase": "per person"},
    792: {"origin": "Delhi", "phrase": "all of us"},
    906: {"origin": "Kolkata", "phrase": "all of us"},
}

CLARIFICATION = (
    "You mentioned travelling solo, but also referred to multiple travellers. "
    "Will you be travelling alone or with others?"
)


def correct_record(record, line):
    """Fail closed if any source example no longer matches the reviewed case."""
    if line not in REVIEWED:
        return record, None
    meta = REVIEWED[line]
    result = copy.deepcopy(record)
    messages = result.get("messages", [])
    user_texts = [
        normalize(msg.get("content", ""))
        for msg in messages if msg.get("role") == "user"
    ]
    if len(user_texts) != 1:
        raise ValueError(f"Line {line}: expected one user prompt")
    user_text = user_texts[0]
    if "solo" not in user_text or meta["phrase"] not in user_text:
        raise ValueError(f"Line {line}: prompt is not the reviewed ambiguous case")
    if normalize(meta["origin"]) not in user_text:
        raise ValueError(f"Line {line}: origin differs from reviewed example")

    assistant_indices = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    if len(assistant_indices) != 1:
        raise ValueError(f"Line {line}: expected exactly one assistant action")
    index = assistant_indices[0]
    content = messages[index].get("content")
    previous = json.loads(content) if isinstance(content, str) else content
    if not isinstance(previous, dict) or previous.get("type") != "state_update":
        raise ValueError(f"Line {line}: expected a state_update action")
    state = previous.get("state")
    if (
        not isinstance(state, dict)
        or state.get("traveller_group") != "solo"
        or state.get("traveller_count") != 1
        or state.get("origin") != meta["origin"]
    ):
        raise ValueError(f"Line {line}: source state differs from reviewed example")
    if "solo_with_plural_reference_review" not in inspect_action(previous, user_text):
        raise ValueError(f"Line {line}: missing expected solo/plural conflict flag")

    updated = {"type": "message", "message": CLARIFICATION}
    messages[index]["content"] = json.dumps(
        updated, ensure_ascii=False, separators=(",", ":")
    )
    change = {
        "jsonl_line": line,
        "user": user_texts[0],
        "before": previous,
        "after": updated,
        "reason": "clarify_solo_vs_plural_request",
    }
    return result, change


def first_question(record):
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return normalize(msg.get("content", ""))
    return None


def eval_question(record):
    if isinstance(record.get("question"), str):
        return normalize(record["question"])
    return first_question(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.source, args.output, args.manifest, args.eval)]
    if len(set(paths)) != len(paths):
        raise ValueError("All input/output/eval paths must differ")
    if not args.source.is_file() or not args.eval.is_file():
        raise FileNotFoundError("Source candidate or eval file missing")

    heldout = {eval_question(row) for _, row in iter_records(args.eval)}
    heldout.discard(None)
    records = []
    changes = []
    seen = set()
    for line, record in iter_records(args.source):
        if first_question(record) in heldout:
            raise ValueError(f"Training/evaluation first-prompt overlap at line {line}")
        fixed, change = correct_record(record, line)
        records.append(fixed)
        if change:
            changes.append(change)
            seen.add(line)
    if seen != set(REVIEWED):
        raise ValueError(f"Missing reviewed lines: {sorted(set(REVIEWED) - seen)}")
    if len(records) != 960:
        raise ValueError(f"Unexpected conversation count: {len(records)}")
    if len(changes) != 5:
        raise ValueError(f"Expected 5 corrections, got {len(changes)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "input_candidate": str(args.source),
        "output_candidate": str(args.output),
        "training_conversations": len(records),
        "reviewed_clarifications": len(changes),
        "overlap_with_given_eval": 0,
        "notes": (
            "Five manually reviewed ambiguous solo/plural training labels changed "
            "to clarification messages. Human review and fresh unseen evaluation "
            "are still needed before claiming generalization."
        ),
        "changes": changes,
    }
    args.manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FINAL SFT V2 CORRECTION CANDIDATE")
    print("Conversations:", len(records))
    print("Changed state updates into clarifications:", len(changes))
    print("Lines changed:", sorted(seen))
    print("Training/eval exact prompt overlaps: 0")
    print("Output:", args.output)
    print("Manifest:", args.manifest)
    print("Source input: UNCHANGED")


if __name__ == "__main__":
    main()
