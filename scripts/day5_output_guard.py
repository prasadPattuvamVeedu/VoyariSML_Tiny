
import json
from day5_schema import validate_action


def reject_duplicate_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_key:{key}")
        result[key] = value

    return result


def check_model_output(raw, user_question):
    try:
        action = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys
        )
    except (ValueError, TypeError) as exc:
        return None, [f"invalid_json:{exc}"]

    if not isinstance(action, dict):
        return None, ["not_json_object"]

    try:
        errors = validate_action(action, user_question)
    except (ValueError, TypeError, KeyError) as exc:
        return None, [f"validation_exception:{exc}"]

    if errors:
        return None, errors

    return action, []
