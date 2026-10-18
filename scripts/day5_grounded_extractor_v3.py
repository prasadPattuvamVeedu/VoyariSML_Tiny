
import copy
import re

from day5_grounded_extractor import extract_grounded as base_extract

PLACE = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"


def normalize(value):
    return " ".join(value.casefold().split())


def add_candidate(result, field, value, evidence):
    """Add evidence without silently replacing conflicting values."""

    if field in result["ambiguous"]:
        options = result["ambiguous"][field]

        if not any(normalize(x) == normalize(value) for x in options):
            options.append(value)

        return

    current = result["values"].get(field)

    if current is None:
        result["values"][field] = {
            "value": value,
            "evidence": evidence,
        }

    elif normalize(current["value"]) != normalize(value):
        result["ambiguous"][field] = [
            current["value"],
            value,
        ]
        del result["values"][field]


def remove_candidate(result, field, unwanted):
    """Remove only a specifically negated candidate."""

    current = result["values"].get(field)

    if current and normalize(current["value"]) == normalize(unwanted):
        del result["values"][field]

    options = result["ambiguous"].get(field)

    if options:
        remaining = [
            value for value in options
            if normalize(value) != normalize(unwanted)
        ]

        del result["ambiguous"][field]

        if len(remaining) == 1:
            result["values"][field] = {
                "value": remaining[0],
                "evidence": remaining[0],
            }

        elif len(remaining) > 1:
            result["ambiguous"][field] = remaining


def prefer_destination(result, value):
    """Resolve only an explicit positive-versus-negative contrast."""

    result["ambiguous"].pop("destination", None)

    result["values"]["destination"] = {
        "value": value,
        "evidence": value,
    }


def extract_grounded_v2(question):
    result = copy.deepcopy(base_extract(question))

    # Additional explicit destination expressions.
    patterns = [
        r"(?i:\bforecast\s+for\s+)" + PLACE,
        r"(?i:\bplace\s+to\s+stay\s+in\s+)" + PLACE,
        r"(?i:\breach\s+)" + PLACE,
        r"(?i:\bsightseeing\s+around\s+)" + PLACE,
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, question):
            add_candidate(
                result,
                "destination",
                match.group(1),
                match.group(1),
            )

    # Conservative lowercase support: single-word candidate only.
    # Destination existence must still be checked against trusted data.
    lowercase_pattern = (
        r"(?i:\btravel\s+to\s+)([a-z][a-z-]*)\b"
    )

    for match in re.finditer(lowercase_pattern, question):
        add_candidate(
            result,
            "destination",
            match.group(1),
            match.group(1),
        )

    # Example: "I'm not departing from Tirur."
    negative_origin = (
        r"(?i:\bnot\s+departing\s+from\s+)" + PLACE
    )

    for match in re.finditer(negative_origin, question):
        remove_candidate(result, "origin", match.group(1))

    # "No hotels in A; search hotels in B."
    contrast_hotel = (
        r"(?i:\bno\s+hotels\s+in\s+)"
        + PLACE
        + r"(?i:\s*;\s*search\s+hotels\s+in\s+)"
        + PLACE
    )

    match = re.search(contrast_hotel, question)

    if match:
        prefer_destination(result, match.group(2))

    # "Visit A, not to B."
    contrast_visit = (
        r"(?i:\bvisit\s+(?:to\s+)?)"
        + PLACE
        + r"(?i:\s*,\s*not\s+to\s+)"
        + PLACE
    )

    match = re.search(contrast_visit, question)

    if match:
        prefer_destination(result, match.group(1))

    # "Not to travel to A, but to visit B."
    contrast_correction = (
        r"(?i:\bnot\s+to\s+travel\s+to\s+)"
        + PLACE
        + r"(?i:\s*,\s*but\s+to\s+visit\s+)"
        + PLACE
    )

    match = re.search(contrast_correction, question)

    if match:
        prefer_destination(result, match.group(2))

    return result
