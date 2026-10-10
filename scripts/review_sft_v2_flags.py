"""Print representative SFT data-audit examples grouped by warning type.

Read-only. Warnings are heuristics, not confirmed labeling errors.
Requires scripts/audit_sft_v2_data.py in the same folder.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_sft_v2_data import inspect_action, iter_records, normalize


def main():
    parser = argparse.ArgumentParser(
        description="Review potential Voyari SFT label problems by warning category"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--per-flag", type=int, default=4)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.per_flag < 1 or args.per_flag > 20:
        raise ValueError("--per-flag must be between 1 and 20")
    if not args.train.is_file():
        raise FileNotFoundError(args.train)

    totals = Counter()
    samples = defaultdict(list)

    for line_no, record in iter_records(args.train):
        history = []
        for message in record.get("messages", []):
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                history.append(str(content))
            if role != "assistant":
                continue

            try:
                action = json.loads(content) if isinstance(content, str) else content
            except (ValueError, TypeError):
                continue

            warnings = inspect_action(action, " ".join(normalize(item) for item in history))
            for flag in warnings:
                totals[flag] += 1
                if len(samples[flag]) < args.per_flag:
                    samples[flag].append({
                        "training_jsonl_line": line_no,
                        "user_history": history[-4:],
                        "assistant_action": action,
                    })

    print("VOYARILM SFT V2 FLAG REVIEW")
    print("Training file:", args.train)
    print("Flag occurrences:", sum(totals.values()))

    report = {"train": str(args.train), "flag_counts": dict(totals), "sampled_examples": {}}
    for flag, count in totals.most_common():
        print("\n" + "=" * 65)
        print("FLAG:", flag, "| Occurrences:", count)
        print("=" * 65)
        report["sampled_examples"][flag] = samples[flag]
        for index, item in enumerate(samples[flag], 1):
            print("\nEXAMPLE", index, "| JSONL line", item["training_jsonl_line"])
            for user in item["user_history"]:
                print("USER:", user[:600])
            print(
                "ASSISTANT:",
                json.dumps(item["assistant_action"], ensure_ascii=False)[:800],
            )

    print("\nThese flags are candidates for manual review, not verified errors.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Grouped review saved:", args.report)


if __name__ == "__main__":
    main()
