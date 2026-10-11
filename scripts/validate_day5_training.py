"""Independent Day 5 label and holdout audit. No GPU, no model updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from day5_schema import parse_action, validate_action
from generate_day5_training import read_user_questions, norm


def validate(path, holdouts=(), other_trains=()):
    forbidden = set()
    for p in (*holdouts, *other_trains):
        forbidden.update(read_user_questions(p))
    seen = set()
    kinds = Counter()
    total = 0
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        total += 1
        record = json.loads(raw)
        messages = record.get("messages", [])
        roles = [m.get("role") for m in messages]
        if roles not in (["system", "user", "assistant"], ["system", "user", "assistant", "user", "assistant"]):
            raise ValueError(f"Line {n}: incorrect roles: {roles}")
        if not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip():
            raise ValueError(f"Line {n}: missing system instruction")
        user = ""
        for msg_index, m in enumerate(messages[1:], start=1):
            if not isinstance(m.get("content"), str):
                raise ValueError(f"Line {n}: message content must be text")
            if m["role"] == "user":
                user = m["content"]
                key = norm(user)
                if key in forbidden or (key in seen and msg_index == 1):
                    raise ValueError(f"Line {n}: duplicate or forbidden evaluation/training question: {user}")
                if msg_index == 1:
                    seen.add(key)
            else:
                action = parse_action(m["content"])
                errors = validate_action(action, user)
                if errors:
                    raise ValueError(f"Line {n}: label errors={errors} input={user} output={action}")
                kinds[action["type"]] += 1
    if total < 1:
        raise ValueError("Empty training file")
    return {"conversations": total, "unique_conversation_starts": len(seen), "assistant_actions": dict(kinds),
            "exact_holdout_or_replay_overlap": 0,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-file", type=Path, required=True)
    p.add_argument("--holdout", type=Path, action="append", default=[])
    p.add_argument("--existing-train", type=Path, action="append", default=[])
    p.add_argument("--expected-count", type=int, default=4000)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    for q in [args.training_file, *args.holdout, *args.existing_train]:
        if not q.is_file():
            raise FileNotFoundError(q)
    result = validate(args.training_file, args.holdout, args.existing_train)
    if result["conversations"] != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count}, got {result['conversations']}")
    result["caution"] = "Passing structural checks does not establish semantic accuracy or generalization. Human spot review required."
    print("DAY 5 DATA VALIDATION PASSED")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Report:", args.report)


if __name__ == "__main__":
    main()
