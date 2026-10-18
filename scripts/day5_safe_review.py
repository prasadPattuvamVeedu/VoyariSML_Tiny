from day5_output_guard import check_model_output
from day5_grounding_audit_v3 import audit_model_output


def review_model_output(question, raw):
    action, schema_errors = check_model_output(
        raw, question
    )

    audit = audit_model_output(question, raw)

    if schema_errors:
        decision = "reject_schema"
    elif audit["grounding_mismatches"]:
        decision = "reject_grounding"
    elif audit["ambiguous_fields"]:
        decision = "block_ambiguous"
    else:
        decision = "manual_review_required"

    return {
        "decision": decision,
        "schema_valid": (
            action is not None and not schema_errors
        ),
        "action_type": (
            action.get("type")
            if isinstance(action, dict)
            else None
        ),
        "tool_execution_allowed": False,
        "state_write_allowed": False,
        "audit": audit,
    }
