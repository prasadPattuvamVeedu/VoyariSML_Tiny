import json
import re

from day5_grounded_extractor_v3 import extract_grounded_v2
from day5_safe_review import review_model_output


def check_clarification(question, raw):
    base = review_model_output(question, raw)

    result = {
        "decision": base["decision"],
        "flags": [],
        "tool_execution_allowed": False,
        "state_write_allowed": False,
    }

    # Existing safety failures must remain failures.
    if base["decision"] != "manual_review_required":
        return result

    action = json.loads(raw)

    if action.get("type") != "message":
        return result

    message = action.get("message", "")
    if not isinstance(message, str):
        return result

    text = message.casefold()
    user = question.casefold()

    extracted = extract_grounded_v2(question)
    values = extracted["values"]

    # Identify which city the model is asking about.
    asks_city = bool(
        re.search(
            r"\b(?:which|what)\s+(?:city|place|destination)\b",
            text,
        )
    )

    asks_origin = asks_city and bool(
        re.search(
            r"\b(?:travelling|traveling|starting|leaving|departing)"
            r"\s+from\b|\borigin\b",
            text,
        )
    )

    asks_destination = asks_city and not asks_origin

    # Do not ask for information already supplied.
    if asks_destination and "destination" in values:
        result["flags"].append(
            "destination_already_provided"
        )

    if asks_origin and "origin" in values:
        result["flags"].append(
            "origin_already_provided"
        )

    # Detect one obvious contradiction.
    if (
        re.search(r"\bsolo\b", user)
        and re.search(r"\btogether\b", user)
    ):
        result["flags"].append(
            "traveller_count_needs_clarification"
        )

    # Booking/package dates are missing, but the
    # model asks about origin instead.
    missing_date_statement = bool(
        re.search(
            r"(?:haven't|have not|aren't|not|undecided)"
            r".{0,55}(?:dates?|when to travel)",
            user,
        )
        or re.search(
            r"(?:dates?|when to travel)"
            r".{0,55}(?:aren't decided|not decided|unknown)",
            user,
        )
    )

    booking_request = bool(
        re.search(
            r"\b(?:book|booking|package|hotels?|accommodation)\b",
            user,
        )
    )

    if (
        asks_origin
        and booking_request
        and missing_date_statement
        and "date_expression" not in values
    ):
        result["flags"].append(
            "review_missing_travel_dates_priority"
        )

    blocking_flags = {
        "destination_already_provided",
        "origin_already_provided",
    }

    if any(f in blocking_flags for f in result["flags"]):
        result["decision"] = "block_redundant_question"

    elif result["flags"]:
        result["decision"] = "manual_clarification_review"

    return result
