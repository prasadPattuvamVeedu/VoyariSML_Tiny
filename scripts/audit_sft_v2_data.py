"""Audit VoyariLM Tiny SFT JSONL for examples requiring manual grounding review.

Read-only, heuristic checks. Understands some date variants, relative duration/
budget arithmetic, and conventional two-person/parent travel groups.
No warnings imply verification, and warnings are not proof of errors.
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
    "tomorrow", "today", "tonight", "weekend", "next week", "this week",
    "next month", "this month", "from now", "depart on", "arrive on",
)
MONTH_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
WEEKDAY_PATTERN = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
UNDECIDED_MARKERS = (
    "undecided", "don't know", "do not know", "haven't decided",
    "not decided", "unknown", "not picked", "not chosen",
    "don't assume", "do not assume", "can't give",
)
NUMERIC_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def normalize(value):
    return re.sub(r"\s+", " ", str(value).casefold().replace("_", " ").strip())


def phrase_present(value, text):
    term = normalize(value)
    if not term:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text))


def has_explicit_date(text):
    """Detect likely travel dates, not bare check-in/check-out words."""
    if any(word in text for word in DATE_MARKERS):
        return True
    if re.search(r"\b(?:next|this|coming)\s+" + WEEKDAY_PATTERN + r"\b", text):
        return True
    if re.search(r"\b" + MONTH_PATTERN + r"\s+\d{1,2}\b", text):
        return True
    if re.search(r"\b(?:in|during)\s+" + MONTH_PATTERN + r"\b", text):
        return True
    if re.search(r"\b\d{1,2}\s+" + MONTH_PATTERN + r"\b", text):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text):
        return True
    return bool(re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text))


def supported_date_expression(date, text):
    """Check date expressions using literal/normalized evidence only."""
    value = normalize(date)
    if phrase_present(value, text):
        return True
    # Recognize common language equivalents without guessing a new date.
    aliases = {
        "next weekend": ("coming weekend", "upcoming weekend"),
        "tomorrow": ("the following day",),
        "today": ("this day",),
    }
    return any(phrase_present(alias, text) for alias in aliases.get(value, ()))


def duration_mentioned(value, text):
    """Recognize explicit trip lengths and simple +/- one day updates."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return True

    def mentioned(x):
        if re.search(r"(?<!\d)" + str(x) + r"\s*[- ]?\s*(?:day|night|week)s?\b", text):
            return True
        words = NUMBER_WORDS.get(x, ())
        return any(
            re.search(r"\b" + re.escape(word) + r"\s*[- ]?\s*(?:day|night|week)s?\b", text)
            for word in words
        )

    if mentioned(n):
        return True
    if (re.search(r"\b(?:add|increase|extend)\s+(?:by\s+)?(?:one|1)\s+(?:more\s+)?day\b", text)
            or "one more day" in text):
        return n > 1 and mentioned(n - 1)
    if re.search(r"\b(?:one|1)\s+day\s+shorter\b", text) or "reduce by one day" in text:
        return mentioned(n + 1)
    return False


def numeric_mentioned(value, text):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return True
    stripped = re.sub(r"(?<=\d),(?=\d)", "", text)
    return bool(re.search(r"(?<!\d)" + str(number) + r"(?!\d)", stripped))


def budget_supported(value, text):
    """Check literal budget or simple currency increase/decrease in a dialogue."""
    if numeric_mentioned(value, text):
        return True
    try:
        desired = int(value)
    except (TypeError, ValueError):
        return True
    # Only parse explicit rupee amounts, avoiding durations and head counts.
    amounts = [
        int(digits.replace(",", ""))
        for digits in re.findall(r"(?:₹|rs\.?\s*)(\d[\d,]*)", text)
    ]
    if len(amounts) < 2:
        return False
    base, delta = amounts[-2:]
    if re.search(r"\b(?:increase|raise|add)\b.{0,40}\bbudget\b|\bbudget\b.{0,30}\b(?:increase|raise|add)\b", text):
        return base + delta == desired
    if re.search(r"\b(?:reduce|lower|decrease|cut)\b.{0,40}\bbudget\b|\bbudget\b.{0,30}\b(?:reduce|lower|decrease|cut)\b", text):
        return base - delta == desired
    return False


def traveller_count_supported(value, text):
    """Allow unambiguous small groups; don't assume group-size defaults."""
    if numeric_mentioned(value, text):
        return True
    try:
        number = int(value)
    except (TypeError, ValueError):
        return True
    if any(
        re.search(r"\b" + word + r"\b", text) and number == n
        for word, n in NUMERIC_WORDS.items()
    ):
        # Numeric words alone are weak evidence; prefer count-bearing phrases.
        if re.search(r"\b(?:family|group|party|friends|people|travellers|travelers|persons|of)\s+(?:of\s+)?(?:"
                     + "|".join(re.escape(w) for w, n in NUMERIC_WORDS.items() if n == number)
                     + r")\b", text):
            return True
    # Context convention: the requester plus partner = two people.
    if number == 2 and re.search(r"\b(?:with my partner|with my spouse|with my wife|with my husband)\b", text):
        return True
    # Two parents plus requester, unless an explicitly different group is given.
    if number == 3 and re.search(r"\bwith my parents\b", text):
        return True
    return False


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
        elif date and not supported_date_expression(date, user_context):
            warnings.append("tool_date_expression_not_verbatim")
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
        if "budget_inr" in state and not budget_supported(state["budget_inr"], user_context):
            warnings.append("budget_not_explicit_in_user_context")
        if "traveller_count" in state and not traveller_count_supported(state["traveller_count"], user_context):
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
