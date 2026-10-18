import re

# Match one- to three-word capitalized place names.
# These are candidates, not verified destinations.
NAME = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"

DAYS = {
    "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    "ten": 10,
}

ORIGIN_PATTERNS = [
    r"(?i:\bstart\s+at\s+)" + NAME + r"\b",
    r"(?i:\bfrom\s+)" + NAME + r"\b",
    r"(?i:\b(?:starting point|departure city|origin)\s*(?:is|:)\s*)" + NAME + r"\b",
]

DESTINATION_PATTERNS = [
    r"(?i:\bfinish\s+in\s+)" + NAME + r"\b",
    r"(?i:\b(?:historical\s+)?monuments\s+in\s+)" + NAME + r"\b",
    r"(?i:\b(?:accommodations?|stays)\s+in\s+)" + NAME + r"\b",
    r"(?i:\bto\s+)" + NAME + r"\b",
    r"(?i:\b(?:visit|visiting|destination\s*(?:is|:)|going\s+to)\s+)" + NAME + r"\b",
    r"(?i:\bset\s+)" + NAME + r"(?i:\s+as\s+(?:the\s+)?destination\b)",
    r"(?i:\bfor\s+the\s+)" + NAME + r"(?i:\s+visit\b)",
    r"(?i:\b(?:weather|hotels?|accommodation|packages?|activities|information)\b(?:(?!\bbudget\b|\bunder\b|\bprice\b|\bRs\b|\bINR\b)[^.!?\n]){0,80}?\b(?:in|at|of|for)\s+)" + NAME + r"\b",
    r"(?i:\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+in\s+)" + NAME + r"\b",
]

BUDGET = re.compile(
    r"(?:₹\s*|\bRs\.?\s*|\bINR\s*)(\d[\d,]*)",
    re.I
)

RUPEE_SUFFIX = re.compile(r"\b(\d[\d,]*)\s+rupees\b", re.I)

DURATION = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+)days?\b",
    re.I
)

DATE = r"\d{4}-\d{2}-\d{2}"


def add_unique(candidates, field, result):
    unique = list(dict.fromkeys(
        item["value"] for item in candidates
    ))

    if len(unique) == 1:
        result["values"][field] = candidates[0]

    elif len(unique) > 1:
        result["ambiguous"][field] = unique


def extract_grounded(question):
    """
    Extract evidence-backed candidates from user text.

    Does not infer intent, verify place existence,
    change conversation state, or execute tools.
    """

    result = {
        "values": {},
        "ambiguous": {},
    }

    # 1. Origin and destination candidates
    groups = [
        ("origin", ORIGIN_PATTERNS),
        ("destination", DESTINATION_PATTERNS),
    ]

    for field, patterns in groups:
        candidates = []

        for pattern in patterns:
            for match in re.finditer(pattern, question):
                # Exclude a destination rejected in this clause.
                if field == "destination":
                    boundary = max(
                        question.rfind(char, 0, match.start())
                        for char in ".!?;"
                    )
                    prefix = question[boundary + 1:match.start()]

                    if re.search(
                        r"(?i)\b(?:don't|do not)\s+(?:look|search|find|check|book)\b",
                        prefix,
                    ):
                        continue

                candidates.append({
                    "value": match.group(1),
                    "evidence": match.group(1),
                })

        add_unique(candidates, field, result)

    # 2. Exact rupee amounts
    budgets = []

    for match in BUDGET.finditer(question):
        amount = int(match.group(1).replace(",", ""))

        budgets.append({
            "value": amount,
            "evidence": match.group(0),
        })

    for match in RUPEE_SUFFIX.finditer(question):
        budgets.append({
            "value": int(match.group(1).replace(",", "")),
            "evidence": match.group(0),
        })

    add_unique(budgets, "budget_inr", result)

    # 3. Trip duration
    durations = []

    for match in DURATION.finditer(question):
        token = match.group(1).lower()

        days = (
            int(token)
            if token.isdigit()
            else DAYS[token]
        )

        durations.append({
            "value": days,
            "evidence": match.group(0),
        })

    add_unique(durations, "duration_days", result)

    # 4. Total versus per-person budget
    per_person = re.search(
        r"\bper\s+person\b",
        question,
        re.I
    )

    total = re.search(
        r"\btotal\s+(?:available\s+)?budget\b",
        question,
        re.I
    )

    if per_person and total:
        result["ambiguous"]["budget_basis"] = [
            "per_person", "total"
        ]

    elif per_person:
        result["values"]["budget_basis"] = {
            "value": "per_person",
            "evidence": per_person.group(0),
        }

    elif total:
        result["values"]["budget_basis"] = {
            "value": "total",
            "evidence": total.group(0),
        }

    # 5. Explicit hotel check-in/check-out dates
    date_cues = [
        ("check_in", "check[ -]?in"),
        ("check_out", "check[ -]?out"),
    ]

    for field, cue in date_cues:
        pattern = (
            r"(?i:\b" + cue +
            r"\s*(?:date\s*)?(?::|is)?\s*)("
            + DATE + r")"
        )

        match = re.search(pattern, question)

        if match:
            result["values"][field] = {
                "value": match.group(1),
                "evidence": match.group(0),
            }

    # 6. Relative dates: preserve wording
    relative_candidates = [
        {
            "value": match.group(1).lower(),
            "evidence": match.group(0),
        }
        for match in re.finditer(
            r"\b(today|tomorrow|next weekend)\b",
            question,
            re.I
        )
    ]

    add_unique(relative_candidates, "date_expression", result)

    return result
