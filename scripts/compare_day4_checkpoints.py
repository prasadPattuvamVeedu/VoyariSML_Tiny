"""Compare frozen SFT v2 and Day 4 model outputs on the same 30 cases.

Read-only. Prints action changes, category metrics, and original model responses
for clarification regressions (including the evaluation heuristic's limits).
Does not train, modify checkpoints, or edit evaluation source files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EXPECTED_BENCHMARK_SHA = (
    "df424614a58505b2853318e23d1cd85f05242e65ca6dbd8f37a0b372ef103eb4"
)


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for index, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            case_id = row["id"]
            if case_id in rows:
                raise ValueError(f"Duplicate case {case_id} in {path}:{index}")
            rows[case_id] = row
    if len(rows) != 30:
        raise ValueError(f"Expected 30 responses in {path}, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare Day 3 vs Day 4 outputs")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--category", choices=("clarify", "state", "tool", "all"),
                        default="clarify")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    old = load(args.baseline)
    new = load(args.candidate)
    if set(old) != set(new):
        raise ValueError("Question IDs do not match between checkpoints")
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.baseline_summary, args.candidate_summary)
    ]
    if not all(
        summary.get("benchmark_sha256") == EXPECTED_BENCHMARK_SHA
        and summary.get("total") == 30 for summary in summaries
    ):
        raise ValueError("Benchmark hash/size differs; comparison is invalid")
    if summaries[0].get("stage") != "sft_v2" or summaries[0].get("step") != 100:
        raise ValueError("Baseline must be SFT v2 step 100")
    if summaries[1].get("stage") != "sft_day4" or summaries[1].get("step") != 25:
        raise ValueError("Candidate must be Day 4 step 25")

    differences = []
    transitions = Counter()
    for case_id in sorted(old):
        a, b = old[case_id], new[case_id]
        if a["question"] != b["question"] or a["category"] != b["category"]:
            raise ValueError(f"Case content mismatch: {case_id}")
        if a["expected"] != b["expected"]:
            raise ValueError(f"Expected action changed for {case_id}")
        key = a["category"]
        transitions[(key, "action_improved" if not a["action_match"] and b["action_match"]
                     else "action_regressed" if a["action_match"] and not b["action_match"]
                     else "action_unchanged")] += 1
        transitions[(key, "exact_improved" if not a["exact_match"] and b["exact_match"]
                     else "exact_regressed" if a["exact_match"] and not b["exact_match"]
                     else "exact_unchanged")] += 1
        differences.append({
            "id": case_id,
            "category": key,
            "question": a["question"],
            "expected": a["expected"],
            "baseline": {
                "predicted": a.get("predicted"),
                "raw": a.get("raw"),
                "action_match": a["action_match"],
                "exact_match": a["exact_match"],
                "topic_match": a.get("topic_match"),
                "unexpected_fields": a.get("unexpected_fields", []),
            },
            "candidate": {
                "predicted": b.get("predicted"),
                "raw": b.get("raw"),
                "action_match": b["action_match"],
                "exact_match": b["exact_match"],
                "topic_match": b.get("topic_match"),
                "unexpected_fields": b.get("unexpected_fields", []),
            },
        })

    print("DAY 4 — MODEL REGRESSION REVIEW")
    print("Frozen baseline: SFT v2 step 100")
    print("Candidate: Day 4 step 25")
    print("Benchmark cases: 30; matching IDs, questions and expected answers")
    for category in ("state", "tool", "clarify"):
        previous = summaries[0]["categories"][category]
        current = summaries[1]["categories"][category]
        print(
            f"{category}: action {previous['action_match']}/{previous['total']} "
            f"-> {current['action_match']}/{current['total']}; "
            f"exact/topic {previous['exact_or_topic_match']}/{previous['total']} "
            f"-> {current['exact_or_topic_match']}/{current['total']}"
        )
        print(
            "  Action improved:", transitions[(category, "action_improved")],
            "| regressed:", transitions[(category, "action_regressed")],
            "| exact/topic improved:", transitions[(category, "exact_improved")],
            "| regressed:", transitions[(category, "exact_regressed")],
        )

    selected = [r for r in differences if args.category in ("all", r["category"])]
    print(f"\nSELECTED ORIGINAL RESPONSES: {args.category} (showing {min(len(selected),args.limit)})")
    for row in selected[:args.limit]:
        a, b = row["baseline"], row["candidate"]
        print("\n---", row["id"], "---")
        print("USER:", row["question"])
        print("EXPECTED:", json.dumps(row["expected"], ensure_ascii=False))
        print("BASELINE:", json.dumps(a["predicted"], ensure_ascii=False))
        print(
            "  action:", a["action_match"], "| exact/topic:", a["exact_match"],
            "| clarification heuristic:", a["topic_match"],
        )
        print("DAY 4:", json.dumps(b["predicted"], ensure_ascii=False))
        print(
            "  action:", b["action_match"], "| exact/topic:", b["exact_match"],
            "| clarification heuristic:", b["topic_match"],
        )
    if args.report:
        if args.report.resolve() in {args.baseline.resolve(), args.candidate.resolve(),
                                     args.baseline_summary.resolve(),
                                     args.candidate_summary.resolve()}:
            raise ValueError("Report must not overwrite results or summaries")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA,
                "baseline": summaries[0], "candidate": summaries[1],
                "category_action_transitions": [
                    {"category": k[0], "outcome": k[1], "count": count}
                    for k, count in sorted(transitions.items())
                ],
                "cases": differences,
                "caution": ("Clarification relevance is keyword-scored; inspect original "
                            "responses manually. All cases are a development diagnostic."),
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("\nFull comparison report:", args.report)


if __name__ == "__main__":
    main()
