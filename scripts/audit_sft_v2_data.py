"""Audit VoyariLM Tiny SFT JSONL for examples that need manual grounding review.

This script does NOT change training data or claim automatically that a label is wrong.
It examines first-user-action examples and supports multi-turn context.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

NUMBER_WORDS = {
    1: ("one", "a day", "single day"),
    2: ("two", "couple", "2-day"),
    3: ("three", "3-day"),
    4: ("four", "4-day"),
    5: ("five", "5-day"),
    6: ("six", "6-day"),
    7: ("seven", "7-day"),
    8: ("eight", "8-day"),
    9: ("nine", "9-day"),
    10: ("ten", "10-day"),
}
DATE_MARKERS = (
    "tomorrow", "today", "tonight", "weekend", "next week",
    "this week", "next month", "this month", "on monday", "on tuesday",
    "on wednesday", "on thursday", "on friday", "on saturday",
    "on sunday", "check-in", "check in", "check-out", "check out",
    "depart on", "arrive on", "in november", "in december",
    "in january", "in february", "in march", "in april", "in may",
    "in june", "in july", "in august", "in september", "in october",
)
UNDECIDED_MARKERS = (
    "undecided", "don't know", "do not know", "haven't decided",
    "not decided", "unknown", "not picked", "not chosen",
    "don't assume", "do not assume", "can't give",
)

def normalize(value):
    return re.sub(r"\s+", " ", str(value).casefold().replace("_", " ").strip())


def phrase_present(value, text):
    term = normalize(value)
    if not term:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text))


def has_explicit_date(text):
    if any(word in text for word in DATE_MARKERS):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text):
        return True
    return bool(re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text))


def duration_mentioned(value, text):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return True
    if re.search(r"(?<!\d)" + str(n) + r"\s*[- ]?\s*(day|night|week)", text):
        return True
    for word in NUMBER_WORDS.get(n, ()):
        if re.search(r"(?<!\w)" + re.escape(word) + r"\s*[- ]?\s*(day|night|week)", text):
            return True
    return False


def numeric_mentioned(value, text):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return True
    stripped = re.sub(r"(?<=\d),(?=\d)", "", text)
    return bool(re.search(r"(?<!\d)" + str(number) + r"(?!\d)", stripped))


def inspect_action(action, user_context):
    """Return weak-signal warnings; every warning requires human verification."""
    warnings = []
    if not isinstance(action, dict):
        return ["non_object_assistant_json"]
    action_type = action.get("type")
    if action_type == "tool_call":
        args = action.get("arguments", {})
        if not isinstance(args, dict):
            return ["tool_arguments_not_object"]
        date = args.get("date_expression") or args.get("date")
        if date and not has_explicit_date(user_context):
            warnings.append("tool_date_without_explicit_date")
        if date and any(w in user_context for w in UNDECIDED_MARKERS):
            if not has_explicit_date(user_context):
                warnings.append("tool_date_despite_undecided_dates")
        location = args.get("location") or args.get("query")
        if isinstance(location, str) and len(location) <= 80:
            if not phrase_present(location, user_context):
                warnings.append("tool_location_not_verbatim_in_user_context")
    elif action_type == "state_update":
        state = action.get("state", {})
        if not isinstance(state, dict):
            return ["state_not_object"]
        for field in ("origin", "destination"):
            value = state.get(field)
            if isinstance(value, str) and not phrase_present(value, user_context):
                warnings.append(field + "_not_verbatim_in_user_context")
        if "duration_days" in state and not duration_mentioned(state["duration_days"], user_context):
            warnings.append("duration_not_explicit_in_user_context")
        if "budget_inr" in state and not numeric_mentioned(state["budget_inr"], user_context):
            warnings.append("budget_not_explicit_in_user_context")
        if "traveller_count" in state and not numeric_mentioned(state["traveller_count"], user_context):
            warnings.append("traveller_count_not_numeric_in_user_context")
        for interest in state.get("interests", []):
            if isinstance(interest, str) and not phrase_present(interest, user_context):
                warnings.append("interest_not_verbatim_in_user_context")
    return sorted(set(warnings))


def iter_records(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            yield line_number, json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    args = parser.parse_args()

    if args.max_examples < 1:
        raise ValueError("--max-examples must be positive")
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    if args.eval and not args.eval.is_file():
        raise FileNotFoundError(args.eval)

    action_counts = Counter()
    warning_counts = Counter()
    report_rows = []
    seen_questions = set()
    conversations = 0
    assistant_turns = 0
    dup_questions = 0

    eval_questions = set()
    if args.eval:
        for _, record in iter_records(args.eval):
            users = [normalize(m.get("content", "")) for m in record.get("messages", [])
                     if m.get("role") == "user"]
            if users:
                eval_questions.add(users[0])
    eval_overlap = 0

    for line, record in iter_records(args.train):
        conversations += 1
        history = []
        messages = record.get("messages", [])
        users = [normalize(m.get("content", "")) for m in messages if m.get("role") == "user"]
        if users:
            if users[0] in seen_questions:
                dup_questions += 1
            seen_questions.add(users[0])
            if users[0] in eval_questions:
                eval_overlap += 1
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                history.append(normalize(content))
            if role != "assistant":
                continue
            assistant_turns += 1
            try:
                action = json.loads(content) if isinstance(content, str) else content
            except (TypeError, ValueError):
                warning_counts["assistant_not_json"] += 1
                if len(report_rows) < args.max_examples:
                    report_rows.append({"line": line, "flags": ["assistant_not_json"],
                                        "user_history": history[-4:],
                                        "assistant": str(content)[:350]})
                continue
            kind = action.get("type", "unknown") if isinstance(action, dict) else "not_object"
            action_counts[kind] += 1
            warnings = inspect_action(action, " ".join(history))
            if warnings:
                warning_counts.update(warnings)
                if len(report_rows) < args.max_examples:
                    report_rows.append({
                        "line": line,
                        "flags": warnings,
                        "user_history": history[-4:],
                        "assistant": action,
                    })

    summary = {
        "training_file": str(args.train),
        "conversation_count": conversations,
        "assistant_turns": assistant_turns,
        "action_counts": dict(action_counts),
        "potential_issue_counts": dict(warning_counts),
        "duplicate_first_user_questions": dup_questions,
        "exact_first_user_eval_overlap": eval_overlap,
        "caution": (
            "Flags are heuristic candidates for human review, not verified errors. "
            "Paraphrases, missing previous trip state, multi-turn context and tool "
            "results can produce false positives. Do not delete flagged records automatically."
        ),
        "examples": report_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("VOYARILM TINY SFT DATA AUDIT")
    print("Conversations:", conversations)
    print("Assistant turns:", assistant_turns)
    print("Action counts:", dict(action_counts))
    print("Potential issues (requires review):")
    for key, value in warning_counts.most_common():
        print(f"  {key}: {value}")
    print("Duplicate first-user prompts:", dup_questions)
    print("Exact evaluation overlap:", eval_overlap)
    print("Example records captured:", len(report_rows))
    print("Report:", args.report)


if __name__ == "__main__":
    main()
