"""Independent deterministic label review for Day 4 synthetic SFT candidate.

- Checks full dataset for literal grounding and exact schema consistency.
- Prints a stratified sample for human inspection.
- Saves a JSON report, NEVER modifies the training JSONL.
This is not a proof of semantic quality; human review remains essential.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from generate_day4_grounded_sft import DAY_WORDS


def flat(value):
    return " ".join(str(value).casefold().split())


def phrase_present(term, user):
    return re.search(r"(?<!\w)" + re.escape(flat(term)) + r"(?!\w)", flat(user)) is not None


def warnings_for(user, action):
    """Independent checks not based on audit_sft_v2_data.inspect_action."""
    issues = []
    kind = action.get("type")
    if kind == "state_update":
        state = action.get("state")
        if not isinstance(state, dict):
            return ["invalid_state"]
        origin = state.get("origin")
        if not isinstance(origin, str) or not phrase_present(origin, user):
            issues.append("origin_not_grounded")
        days = state.get("duration_days")
        if not isinstance(days, int) or isinstance(days, bool) or days not in DAY_WORDS:
            issues.append("invalid_duration")
        else:
            day_word = DAY_WORDS[days]
            if not (
                re.search(r"(?<!\d)" + str(days) + r"\s+days?\b", flat(user))
                or re.search(r"\b" + day_word + r"\s+days?\b", flat(user))
            ):
                issues.append("duration_not_grounded")
        has_budget = "budget_inr" in state
        has_basis = "budget_basis" in state
        if has_budget != has_basis:
            issues.append("budget_value_basis_mismatch")
        if has_budget:
            amount = state["budget_inr"]
            if not isinstance(amount, int) or isinstance(amount, bool):
                issues.append("invalid_budget")
            elif not re.search(
                r"₹\s*" + re.escape(f"{amount:,}") + r"(?!\d)",
                user,
            ):
                issues.append("budget_not_literal")
            basis = state["budget_basis"]
            if basis not in ("total", "per_person"):
                issues.append("invalid_budget_basis")
            elif basis == "per_person" and not re.search(
                r"\bper.person\b|\beach.traveller\b", flat(user)
            ):
                issues.append("per_person_not_grounded")
            elif basis == "total" and not re.search(
                r"\btotal\b|\baltogether\b|\bfull.trip\b", flat(user)
            ):
                issues.append("total_budget_not_grounded")
        else:
            if re.search(r"₹\s*\d", user):
                issues.append("present_budget_omitted")

        has_interests = "interests" in state
        if has_interests:
            interests = state["interests"]
            if not isinstance(interests, list) or len(interests) != 1:
                issues.append("invalid_interests")
            elif not phrase_present(interests[0], user):
                issues.append("interest_not_grounded")
        if "date_expression" in state or "destination" in state:
            issues.append("invented_date_or_destination")
        count = state.get("traveller_count")
        group = state.get("traveller_group")
        expected_count = {"solo": 1, "couple": 2, "family_parents": 3}
        if group in ("family", "friends") and count is not None:
            issues.append("assumed_unknown_group_size")
        if group in expected_count and count != expected_count[group]:
            issues.append("unexpected_group_count")
        if count is not None and group not in expected_count:
            issues.append("group_count_without_explicit_support")
        cues = {
            "solo": ("solo", "alone", "just me"),
            "couple": ("partner", "spouse"),
            "family_parents": ("my parents",),
            "friends": ("friends",),
            "family": ("family",),
        }
        if group is not None and (
            group not in cues or not any(phrase_present(cue, user) for cue in cues[group])
        ):
            issues.append("traveller_group_not_grounded")
        if group is None and any(phrase_present(cue, user)
                                 for group_cues in cues.values() for cue in group_cues):
            # no_group text can say travellers/group but not the above actual cues
            issues.append("explicit_traveller_group_omitted")
    elif kind == "tool_call":
        tool = action.get("tool")
        args = action.get("arguments")
        if not isinstance(args, dict):
            return ["invalid_tool_arguments"]
        if tool == "weather":
            if set(args) != {"location", "date_expression"}:
                issues.append("weather_schema")
            if not phrase_present(args.get("location", ""), user):
                issues.append("weather_location_not_grounded")
            if not phrase_present(args.get("date_expression", ""), user):
                issues.append("weather_date_not_grounded")
        elif tool == "destination_knowledge":
            if set(args) != {"query"}:
                issues.append("destination_knowledge_schema")
            if not phrase_present(args.get("query", ""), user):
                issues.append("knowledge_query_not_grounded")
        else:
            issues.append("unexpected_tool")
    elif kind == "message":
        message = action.get("message")
        if not isinstance(message, str) or "?" not in message:
            issues.append("message_not_question")
        else:
            u = flat(user)
            a = flat(message)
            if any(t in u for t in ("hotel", "stay", "rooms", "accommodation")):
                if not any(t in a for t in ("date", "check-in", "check-out", "when")):
                    issues.append("hotel_dates_not_requested")
            elif any(t in u for t in ("solo trip", "going alone", "travelling solo", "said solo", "start from")):
                if not any(t in a for t in ("alone", "group", "how many", "traveller", "traveler")):
                    issues.append("contradiction_not_clarified")
            elif any(t in u for t in ("transport", "buses", "travel routes", "options")):
                if not any(t in a for t in ("depart", "start", "origin", "from")):
                    issues.append("missing_origin_not_clarified")
    else:
        issues.append("unknown_action")
    return issues


def main():
    p = argparse.ArgumentParser(description="Review grounded Day 4 training labels")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sample-state", type=int, default=12)
    p.add_argument("--sample-tool", type=int, default=6)
    p.add_argument("--sample-clarify", type=int, default=6)
    p.add_argument("--seed", type=int, default=2041)
    args = p.parse_args()
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    if args.report.resolve() == args.train.resolve():
        raise ValueError("Report cannot overwrite training file")
    if min(args.sample_state, args.sample_tool, args.sample_clarify) < 0:
        raise ValueError("Sample counts must be nonnegative")

    cases = {"state_update": [], "tool_call": [], "message": []}
    issues = Counter()
    issue_cases = []
    with args.train.open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            messages = record["messages"]
            user = messages[1]["content"]
            action = json.loads(messages[2]["content"])
            kind = action.get("type")
            if kind not in cases:
                raise ValueError(f"Unrecognized label type at line {line_no}: {kind}")
            item = {"line": line_no, "category": kind,
                    "user": user, "assistant": action}
            flags = warnings_for(user, action)
            if flags:
                issues.update(flags)
                issue_cases.append({**item, "warnings": flags})
            cases[kind].append(item)

    rng = random.Random(args.seed)
    requests = {
        "state_update": args.sample_state,
        "tool_call": args.sample_tool,
        "message": args.sample_clarify,
    }
    samples = []
    for kind, count in requests.items():
        if count > len(cases[kind]):
            raise ValueError(f"Requested {count} samples; only {len(cases[kind])} {kind}")
        samples.extend(rng.sample(cases[kind], count))
    report = {
        "source": str(args.train),
        "total": sum(len(v) for v in cases.values()),
        "category_counts": {k: len(v) for k, v in cases.items()},
        "independent_review_warning_counts": dict(issues),
        "flagged_examples": issue_cases[:100],
        "review_samples": samples,
        "note": (
            "The full-file rules and sampled cases are a quality gate, not proof "
            "of semantic correctness. Manually inspect labels before training."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    print("DAY 4 — INDEPENDENT LABEL REVIEW")
    print("Total conversations:", report["total"])
    print("Categories:", report["category_counts"])
    print("Independent rule warnings:", dict(issues))
    print("Cases flagged:", len(issue_cases))
    for entry in samples:
        print("\nLINE", entry["line"], "|", entry["category"])
        print("USER:", entry["user"])
        print("EXPECTED:", json.dumps(entry["assistant"], ensure_ascii=False))
    print("\nReview report:", args.report)
    if issue_cases:
        print("\nFIRST FLAGGED EXAMPLES:")
        for item in issue_cases[:5]:
            print(f"LINE {item['line']}:", ", ".join(item["warnings"]))
        raise SystemExit("Review warnings found; inspect report before training.")


if __name__ == "__main__":
    main()
