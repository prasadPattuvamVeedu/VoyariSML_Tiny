from pathlib import Path
import json


ROOT = Path(
    r"D:\voyari_sml\data\VoyariLM_DATA\06_FINAL_TRAINING"
)

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<eot>",
]


def audit_text_file(path):
    print("\n" + "=" * 80)
    print("FILE:", path.name)
    print("=" * 80)

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    lines = text.splitlines()

    non_empty_lines = [
        line for line in lines
        if line.strip()
    ]

    blank_lines = len(lines) - len(non_empty_lines)

    print("Characters       :", len(text))
    print("Total lines      :", len(lines))
    print("Non-empty lines  :", len(non_empty_lines))
    print("Blank lines      :", blank_lines)

    print("\nSpecial-token occurrences:")

    for token in SPECIAL_TOKENS:
        print(
            f"{token:12} : {text.count(token)}"
        )

    period_end = 0
    question_end = 0
    exclamation_end = 0
    other_end = 0

    for line in non_empty_lines:
        line = line.rstrip()

        if line.endswith("."):
            period_end += 1

        elif line.endswith("?"):
            question_end += 1

        elif line.endswith("!"):
            exclamation_end += 1

        else:
            other_end += 1

    print("\nNon-empty line endings:")
    print("Ends with .      :", period_end)
    print("Ends with ?      :", question_end)
    print("Ends with !      :", exclamation_end)
    print("Other ending     :", other_end)

    print("\nFirst 3 non-empty lines:")

    for line in non_empty_lines[:3]:
        print(repr(line[:250]))

    print("\nLast 3 non-empty lines:")

    for line in non_empty_lines[-3:]:
        print(repr(line[:250]))


def scan_strings(value, counts):
    if isinstance(value, str):

        for token in SPECIAL_TOKENS:
            counts[token] += value.count(token)

    elif isinstance(value, dict):

        for item in value.values():
            scan_strings(item, counts)

    elif isinstance(value, list):

        for item in value:
            scan_strings(item, counts)


def audit_jsonl(path):
    print("\n" + "=" * 80)
    print("FILE:", path.name)
    print("=" * 80)

    counts = {
        token: 0
        for token in SPECIAL_TOKENS
    }

    rows = 0
    malformed = 0

    role_counts = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            rows += 1

            try:
                obj = json.loads(line)

            except json.JSONDecodeError:
                malformed += 1
                continue

            scan_strings(obj, counts)

            messages = obj.get("messages", [])

            if isinstance(messages, list):

                for message in messages:

                    if isinstance(message, dict):

                        role = message.get("role")

                        if role:
                            role_counts[role] = (
                                role_counts.get(role, 0) + 1
                            )

    print("JSONL rows       :", rows)
    print("Malformed rows   :", malformed)

    print("\nRoles:")

    for role, count in sorted(role_counts.items()):
        print(f"{role:15} : {count}")

    print("\nLiteral special-token occurrences:")

    for token in SPECIAL_TOKENS:
        print(f"{token:12} : {counts[token]}")


print("VOYARILM TINY — TRAINING BOUNDARY AUDIT")

pretraining = ROOT / "pretraining"

for path in sorted(pretraining.glob("*.txt")):
    audit_text_file(path)


instruction = ROOT / "instruction"

for path in sorted(instruction.glob("*.jsonl")):
    audit_jsonl(path)