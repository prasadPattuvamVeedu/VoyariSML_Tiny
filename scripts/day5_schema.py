"""VoyariLM Tiny Day 5: strict action contract for TRAINING LABELS.

A valid label is not the same as a trustworthy live model output. Django must
revalidate model proposals and never execute tool calls without permission.
"""
from __future__ import annotations

import json
import re

SYSTEM_TEXT = (
    "You are VoyariLM, a travel-focused conversational assistant and controller. "
    "Understand travel intent and constraints, ask concise clarification questions "
    "when important information is missing, update structured trip state, and use "
    "only allowlisted travel tools for current, location-dependent, or externally "
    "verified information. Never invent live prices, weather, routes, availability, "
    "exchange rates, or travel advisories. Return valid JSON only. Allowed action "
    "types: message, state_update, tool_call."
)

STATE_FIELDS = {
    "origin", "destination", "duration_days", "budget_inr", "budget_basis",
    "traveller_group", "traveller_count", "interests",
}
# Tool names are proposed contracts; actual Django providers are future work.
TOOL_FIELDS = {
    "weather": ({"location", "date_expression"}, {"location", "date_expression"}),
    "destination_knowledge": ({"query"}, {"query"}),
    "search_travel_packages": ({"destination"}, {"destination", "duration_days", "max_budget_inr", "origin", "date_expression", "budget_basis"}),
    "search_hotels": ({"destination", "check_in", "check_out"}, {"destination", "check_in", "check_out", "traveller_count"}),
    "search_transport": ({"origin", "destination"}, {"origin", "destination", "date_expression"}),
    "search_activities": ({"destination"}, {"destination", "interest"}),
}
BUDGET_RE = re.compile(r"(?:₹|\bRs\.?\s*)(\s*\d[\d,]*)", re.I)
DAY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def matches_amount(value: int, text: str) -> bool:
    return value in {int(x.replace(",", "").strip()) for x in BUDGET_RE.findall(text)}


def matches_duration(value: int, text: str) -> bool:
    t = text.casefold()
    if re.search(rf"(?<!\d){value}\s*[- ]?\s*days?\b", t):
        return True
    return any(n == value and re.search(rf"\b{word}\s+days?\b", t)
               for word, n in DAY_WORDS.items())


def mentions(value: str, text: str) -> bool:
    return bool(value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I))


def matches_city_role(role: str, value: str, text: str) -> bool:
    """Conservative cue check for generated labels, not a universal NER parser."""
    name = re.escape(value)
    if role == "origin":
        prefix = r"(?:\bfrom|\bleaving\s+from|\bdepart(?:ing)?\s+from|\bstarting\s+(?:in|from)|\bstarting\s+city\s+is|\borigin\s+(?:is|as)|\bdepart\s+from)"
        return bool(re.search(prefix + r"\s+" + name + r"\b", text, re.I))
    prefix = r"(?:\bto|\bvisit|\bvisiting|\bin|\bdestination\s+(?:is|as))"
    return bool(re.search(prefix + r"\s+" + name + r"\b", text, re.I) or
                re.search(name + r"\s+as\s+my\s+destination\b", text, re.I))


def validate_action(action: dict, current_user: str, *, previous_user: str = "") -> list[str]:
    """Check schema and explicit grounding. Template-driven, not general NLU."""
    errors = []
    if not isinstance(action, dict):
        return ["action_not_object"]
    kind = action.get("type")
    if kind == "state_update":
        if set(action) != {"type", "state"} or not isinstance(action.get("state"), dict):
            return ["bad_state_envelope"]
        state = action["state"]
        if not state or set(state) - STATE_FIELDS:
            errors.append("empty_or_unknown_state_fields")
        for key in ("origin", "destination"):
            if key in state and (not isinstance(state[key], str) or not mentions(state[key], current_user)):
                errors.append(key + "_not_grounded")
        for key in ("origin", "destination"):
            if key in state and isinstance(state[key], str) and not matches_city_role(key, state[key], current_user):
                errors.append(key + "_role_cue_missing")
        if "origin" in state and "destination" in state and state["origin"].casefold() == state["destination"].casefold():
            errors.append("same_origin_destination")
        if "duration_days" in state:
            v = state["duration_days"]
            if type(v) is not int or not 1 <= v <= 30 or not matches_duration(v, current_user):
                errors.append("duration_not_grounded")
        if "budget_inr" in state:
            v = state["budget_inr"]
            if type(v) is not int or v <= 0 or not matches_amount(v, current_user):
                errors.append("budget_not_grounded")
        if "budget_basis" in state:
            basis = state["budget_basis"]
            if "budget_inr" not in state or basis not in ("total", "per_person"):
                errors.append("bad_budget_basis")
            elif basis == "per_person" and not re.search(r"\b(?:per.person|each person|each traveller)\b", current_user, re.I):
                errors.append("per_person_not_grounded")
            elif basis == "total" and not re.search(r"\b(?:total|altogether|overall)\b", current_user, re.I):
                errors.append("total_basis_not_grounded")
        if "traveller_count" in state:
            n = state["traveller_count"]
            if type(n) is not int or n < 1 or not (re.search(rf"\b{n}\s+(?:travellers|travelers|people|guests)\b", current_user, re.I) or (n == 1 and re.search(r"\b(?:solo|alone)\b", current_user, re.I))):
                errors.append("traveller_count_not_grounded")
        if "traveller_group" in state:
            group = state["traveller_group"]
            cues = {"solo": r"\b(?:solo|alone)\b", "friends": r"\bfriends\b", "family": r"\bfamily\b", "couple": r"\b(?:partner|spouse|couple)\b"}
            if group not in cues or not re.search(cues.get(group, r"a^"), current_user, re.I):
                errors.append("traveller_group_not_grounded")
        if "interests" in state:
            interests = state["interests"]
            if not isinstance(interests, list) or not interests or not all(isinstance(x, str) and mentions(x, current_user) for x in interests):
                errors.append("interests_not_grounded")
        return errors
    if kind == "message":
        if set(action) != {"type", "message"} or not isinstance(action.get("message"), str) or "?" not in action["message"]:
            errors.append("bad_clarification")
        return errors
    if kind == "tool_call":
        if set(action) != {"type", "tool", "arguments"} or action.get("tool") not in TOOL_FIELDS or not isinstance(action.get("arguments"), dict):
            return ["bad_tool_envelope"]
        tool = action["tool"]
        args = action["arguments"]
        required, allowed = TOOL_FIELDS[tool]
        if not required.issubset(args) or set(args) - allowed:
            errors.append("invalid_tool_arguments")
        for key in ("location", "destination", "origin", "query", "interest"):
            if key in args and (not isinstance(args[key], str) or not mentions(args[key], current_user)):
                errors.append(key + "_tool_not_grounded")
        for key in ("max_budget_inr",):
            if key in args and (type(args[key]) is not int or not matches_amount(args[key], current_user)):
                errors.append("tool_budget_not_grounded")
        if "budget_basis" in args:
            basis = args["budget_basis"]
            if "max_budget_inr" not in args or basis not in ("total", "per_person"):
                errors.append("bad_tool_budget_basis")
            elif basis == "per_person" and not re.search(r"\b(?:per.person|each person|each traveller)\b", current_user, re.I):
                errors.append("tool_per_person_not_grounded")
            elif basis == "total" and not re.search(r"\b(?:total|altogether|overall)\b", current_user, re.I):
                errors.append("tool_total_not_grounded")
        if "duration_days" in args and (type(args["duration_days"]) is not int or not matches_duration(args["duration_days"], current_user)):
            errors.append("tool_duration_not_grounded")
        if "date_expression" in args and (not isinstance(args["date_expression"], str) or not mentions(args["date_expression"], current_user)):
            errors.append("tool_date_not_grounded")
        for key in ("check_in", "check_out"):
            if key in args and (not isinstance(args[key], str) or args[key] not in DATE_RE.findall(current_user)):
                errors.append(key + "_not_grounded")
        if "traveller_count" in args and (type(args["traveller_count"]) is not int or not re.search(rf"\b{args['traveller_count']}\s+(?:travellers|travelers|guests|people)\b", current_user, re.I)):
            errors.append("tool_travellers_not_grounded")
        return errors
    return ["unknown_action_type"]


def parse_action(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("assistant action must be a JSON object")
    return value
