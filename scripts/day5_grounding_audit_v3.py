
import json

from day5_output_guard import (
    check_model_output,
    reject_duplicate_keys,
)
from day5_grounded_extractor_v3 import extract_grounded_v2 as extract_grounded


FIELD_MAP = {
    "origin": "origin",
    "destination": "destination",
    "location": "destination",
    "duration_days": "duration_days",
    "budget_inr": "budget_inr",
    "max_budget_inr": "budget_inr",
    "budget_basis": "budget_basis",
    "date_expression": "date_expression",
    "check_in": "check_in",
    "check_out": "check_out",
}


def normalize(value):
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return value


def audit_model_output(question, raw):
    _, schema_errors = check_model_output(raw, question)
    grounded = extract_grounded(question)

    report = {
        "status": "needs_intent_review",
        "schema_errors": list(schema_errors),
        "grounding_mismatches": [],
        "ambiguous_fields": [],
        "verified_fields": [],
        "unverified_fields": [],
    }

    try:
        action = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (ValueError, TypeError):
        report["status"] = "rejected_schema"
        return report

    if not isinstance(action, dict):
        report["status"] = "rejected_schema"
        return report

    action_type = action.get("type")

    if action_type == "state_update":
        payload = action.get("state", {})
    elif action_type == "tool_call":
        payload = action.get("arguments", {})
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    for field, predicted in payload.items():
        source_field = FIELD_MAP.get(field)

        if source_field is None:
            continue

        # A destination_knowledge query may contain a longer
        # description, not just a location. Do not guess.
        if field == "location" and action_type != "tool_call":
            continue

        ambiguous = grounded["ambiguous"].get(source_field)

        if ambiguous is not None:
            report["ambiguous_fields"].append({
                "field": field,
                "candidates": ambiguous,
            })
            continue

        evidence = grounded["values"].get(source_field)

        if evidence is None:
            report["unverified_fields"].append(field)
            continue

        expected = evidence["value"]

        if normalize(predicted) == normalize(expected):
            report["verified_fields"].append(field)
        else:
            report["grounding_mismatches"].append({
                "field": field,
                "model_value": predicted,
                "user_value": expected,
                "evidence": evidence["evidence"],
            })

    if schema_errors:
        report["status"] = "rejected_schema"

    elif report["grounding_mismatches"]:
        report["status"] = "rejected_grounding"

    elif report["ambiguous_fields"]:
        report["status"] = "blocked_ambiguous"

    # Otherwise remains needs_intent_review.
    # This function never authorizes execution.

    return report
