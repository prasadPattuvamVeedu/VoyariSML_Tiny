"""Create a reviewed-candidate SFT JSONL without mutating the original data.

Only apply conservative, rule-based changes:
* Hotel availability that silently invents next_weekend -> date clarification.
* Unsupported headcount of 4 with an unspecified friends/family group -> omit count.

Other cases are retained and listed for manual review. A diff manifest is written.
No training, checkpoint, or evaluation files are modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from audit_sft_v2_data import (
    has_explicit_date,
    inspect_action,
    iter_records,
    normalize,
)


def first_user_text(messages):
    return next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "user"),
        "",
    )


def user_context(messages, assistant_index):
    return " ".join(
        normalize(m.get("content", ""))
        for m in messages[:assistant_index]
        if m.get("role") == "user"
    )


def propose(action, context):
    """Return (new_action, changes) only for narrowly supported corrections."""
    proposed = copy.deepcopy(action)
    changes = []
    warnings = set(inspect_action(action, context))

    if (
        "tool_date_without_explicit_date" in warnings
        and action.get("type") == "tool_call"
        and action.get("tool") == "accommodation_search"
        and isinstance(action.get("arguments"), dict)
    ):
        args = action["arguments"]
        place = args.get("location")
        date = normalize(args.get("date_expression", ""))
        if (
            date == "next weekend"
            and isinstance(place, str)
            and 0 < len(place) <= 80
            and not has_explicit_date(context)
            and (
                "hotel" in context or "stay" in context
                or "accommodation" in context
            )
        ):
            proposed = {
                "type": "message",
                "message": (
                    f"What check-in and check-out dates are you considering "
                    f"for your stay in {place}?"
                ),
            }
            changes.append("replace_invented_hotel_date_with_clarification")

    if (
        "traveller_count_not_numeric_in_user_context" in warnings
        and action.get("type") == "state_update"
        and isinstance(action.get("state"), dict)
    ):
        state = action["state"]
        group = state.get("traveller_group")
        count = state.get("traveller_count")
        # Guard against deleting specified counts or calculable headcounts.
        # The exact wording is intentionally restrictive.
        generic_group = bool(
            re.search(
                r"\b(?:with(?: my)?|going with|travelling with|traveling with)\s+"
                r"(?:my\s+)?(?:friends|family)\b",
                context,
            )
        )
        explicit_party_size = bool(
            re.search(
                r"\b(?:party|group|family)\s+of\s+(?:\d+|one|two|three|four|five|six)\b"
                r"|\b\d+\s+(?:friends|people|travellers|travelers|members)\b"
                r"|\b(?:one|two|three|four|five|six)\s+"
                r"(?:friends|people|travellers|travelers|members)\b",
                context,
            )
        )
        if (
            group in ("friends", "family")
            and count == 4
            and generic_group
            and not explicit_party_size
        ):
            proposed["state"].pop("traveller_count", None)
            changes.append("remove_unsubstantiated_group_size_default")

    return proposed, changes


def validate_record(record):
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Record missing nonempty messages array")
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        action = json.loads(content) if isinstance(content, str) else content
        if not isinstance(action, dict):
            raise ValueError("Assistant content must be a JSON object")
        if action.get("type") not in ("message", "state_update", "tool_call"):
            raise ValueError("Unsupported assistant action type")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval", type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    paths = [args.source.resolve(), args.output.resolve(), args.manifest.resolve()]
    if len(set(paths)) != 3:
        raise ValueError("Source, output and manifest must be different paths")
    if args.eval and not args.eval.is_file():
        raise FileNotFoundError(args.eval)

    heldout = set()
    if args.eval:
        for _, record in iter_records(args.eval):
            heldout.add(normalize(first_user_text(record["messages"])))

    output_records = []
    corrections = []
    counts = Counter()
    remaining = Counter()
    input_count = 0
    eval_overlap = 0

    for line_number, record in iter_records(args.source):
        input_count += 1
        validate_record(record)
        copy_record = copy.deepcopy(record)
        messages = copy_record["messages"]
        if normalize(first_user_text(messages)) in heldout:
            eval_overlap += 1

        for index, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            before_text = msg.get("content")
            action = json.loads(before_text) if isinstance(before_text, str) else before_text
            context = user_context(messages, index)
            after, reasons = propose(action, context)
            if reasons:
                msg["content"] = json.dumps(
                    after, ensure_ascii=False, separators=(",", ":")
                )
                for reason in reasons:
                    counts[reason] += 1
                corrections.append({
                    "line": line_number,
                    "assistant_message_index": index,
                    "reasons": reasons,
                    "user_context": context,
                    "before": action,
                    "after": after,
                })
            remaining.update(inspect_action(after, context))
        validate_record(copy_record)
        output_records.append(copy_record)

    if eval_overlap:
        raise ValueError(
            f"{eval_overlap} exact first-user question overlaps held-out evaluation"
        )
    if input_count != len(output_records):
        raise RuntimeError("Conversation count changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(args.source),
        "candidate_output": str(args.output),
        "conversations": input_count,
        "changed_assistant_turns": len(corrections),
        "changes_by_reason": dict(counts),
        "remaining_heuristic_warnings": dict(remaining),
        "exact_heldout_question_overlap": eval_overlap,
        "notes": (
            "This is a candidate dataset, not production-approved data. "
            "Review all changes and remaining warnings before training. "
            "The source file is unchanged."
        ),
        "changes": corrections,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("VOYARILM TINY — CORRECTIVE DATA CANDIDATE")
    print("Source:", args.source)
    print("Conversations:", input_count)
    print("Corrected assistant turns:", len(corrections))
    print("Changes:", dict(counts))
    print("Remaining review warnings:", dict(remaining))
    print("Held-out exact question overlap:", eval_overlap)
    print("Candidate:", args.output)
    print("Change report:", args.manifest)
    print("Original source: UNCHANGED")


if __name__ == "__main__":
    main()
