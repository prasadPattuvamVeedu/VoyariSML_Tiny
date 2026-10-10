"""Analyze Day 4 model outputs to identify missing, incorrect and invented fields.

Read-only for model outputs and frozen benchmark. Prints the original user
question, expected fields and prediction for manual review. No model inference
and no training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def normalize(value):
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def compare_fields(gold, predicted):
    """Only compare fields once the correct action type is selected."""
    missing = sorted(set(gold) - set(predicted))
    extra = sorted(set(predicted) - set(gold))
    incorrect = {
        key: {"expected": gold[key], "predicted": predicted[key]}
        for key in sorted(set(gold) & set(predicted))
        if normalize(gold[key]) != normalize(predicted[key])
    }
    return missing, extra, incorrect


def analyze(row):
    category = row["category"]
    expected = row["expected"]
    predicted = row.get("predicted")
    result = {
        "id": row["id"],
        "category": category,
        "question": row["question"],
        "expected": expected,
        "predicted": predicted,
        "valid_json": row["valid_json"],
        "action_match": row["action_match"],
        "exact_match": row["exact_match"],
        "missing_fields": [],
        "extra_fields": [],
        "incorrect_fields": {},
        "wrong_tool": None,
        "clarification_on_topic": row.get("topic_match"),
    }
    if not isinstance(predicted, dict) or not row["action_match"]:
        return result
    if category in ("state", "tool"):
        key = "state" if category == "state" else "arguments"
        gold = expected.get(key)
        actual = predicted.get(key)
        if isinstance(gold, dict) and isinstance(actual, dict):
            missing, extra, wrong = compare_fields(gold, actual)
            result["missing_fields"] = missing
            result["extra_fields"] = extra
            result["incorrect_fields"] = wrong
        else:
            result["missing_fields"] = sorted(gold) if isinstance(gold, dict) else []
        if category == "tool" and predicted.get("tool") != expected.get("tool"):
            result["wrong_tool"] = predicted.get("tool")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--category", choices=("state", "tool", "clarify", "all"),
                        default="state")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(args.results)
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    with args.results.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 30:
        raise ValueError(f"Expected 30 Day 4 responses, got {len(rows)}")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate result ID: {row['id']}")
        seen.add(row["id"])

    selected = [
        analyze(row) for row in rows
        if args.category in ("all", row["category"])
    ]
    missing, wrong, extra = Counter(), Counter(), Counter()
    wrong_actions = 0
    exact = 0
    for item in selected:
        if not item["action_match"]:
            wrong_actions += 1
        if item["exact_match"]:
            exact += 1
        missing.update(item["missing_fields"])
        wrong.update(item["incorrect_fields"].keys())
        extra.update(item["extra_fields"])

    summary = {
        "results_file": str(args.results),
        "category": args.category,
        "tested": len(selected),
        "wrong_actions": wrong_actions,
        "exact_or_topic_match": exact,
        "missing_fields": dict(missing),
        "incorrect_fields": dict(wrong),
        "invented_extra_fields": dict(extra),
        "caution": (
            "Structural comparison detects exact missing/wrong/extra fields but "
            "not all semantic mistakes; clarification topics still need human review."
        ),
    }
    print("DAY 4 — FAILURE BREAKDOWN")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for item in selected[:args.limit]:
        print("\n" + "-" * 60)
        print("CASE", item["id"], "|", item["category"])
        print("USER:", item["question"])
        print("EXPECTED:", json.dumps(item["expected"], ensure_ascii=False))
        print("PREDICTED:", json.dumps(item["predicted"], ensure_ascii=False))
        print("ACTION OK:", item["action_match"], "| FULLY CORRECT:", item["exact_match"])
        if item["category"] != "clarify":
            print("MISSING:", item["missing_fields"])
            print("WRONG:", json.dumps(item["incorrect_fields"], ensure_ascii=False))
            print("EXTRA:", item["extra_fields"])
            if item["wrong_tool"] is not None:
                print("WRONG TOOL:", item["wrong_tool"])
        else:
            print("CLARIFICATION TOPIC OK:", item["clarification_on_topic"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "summary": summary, "cases": selected,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("\nReview report:", args.report)


if __name__ == "__main__":
    main()
