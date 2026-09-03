import hashlib
import json
import re
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\voyari_sml")

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
)

INSTRUCTION_ROOT = (
    DATA_ROOT
    / "04_INSTRUCTION"
)

OUTPUT_ROOT = (
    INSTRUCTION_ROOT
    / "normalized_candidates"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT DATASETS
# ============================================================

INDIA_ITINERARIES_PATH = (
    INSTRUCTION_ROOT
    / "india_travel_itineraries"
    / "india_travel_itineraries.jsonl"
)

MULTIWOZ_PATH = (
    INSTRUCTION_ROOT
    / "multiwoz_2_2"
    / "multiwoz_2_2_train.jsonl"
)

BITEXT_PATH = (
    INSTRUCTION_ROOT
    / "bitext_travel"
    / "bitext_travel_train.jsonl"
)

TASKMASTER_PATH = (
    INSTRUCTION_ROOT
    / "taskmaster_2"
    / "taskmaster_2_travel.jsonl"
)

TRAVELPLANNER_PATH = (
    INSTRUCTION_ROOT
    / "travelplanner"
    / "train.jsonl"
)


# ============================================================
# 3. BASIC TEXT CLEANER
# ============================================================

def clean_text(value):
    """
    Make text consistent without changing its meaning.
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    cleaned_lines = []

    for line in text.split("\n"):

        line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(
        cleaned_lines
    ).strip()


# ============================================================
# 4. SPEAKER → VOYARI ROLE
# ============================================================

def normalize_role(speaker):

    if speaker is None:
        return None

    # MultiWOZ sometimes represents speakers numerically.
    if isinstance(
        speaker,
        int,
    ):

        if speaker == 0:
            return "user"

        if speaker == 1:
            return "assistant"

    speaker = str(
        speaker
    ).strip().upper()

    if speaker in {
        "USER",
        "HUMAN",
        "CUSTOMER",
        "0",
    }:
        return "user"

    if speaker in {
        "ASSISTANT",
        "SYSTEM",
        "AGENT",
        "BOT",
        "1",
    }:
        return "assistant"

    return None


# ============================================================
# 5. MERGE CONSECUTIVE SAME-ROLE MESSAGES
# ============================================================

def merge_adjacent_messages(messages):

    merged = []

    for message in messages:

        role = message.get(
            "role"
        )

        content = clean_text(
            message.get(
                "content",
                "",
            )
        )

        if (
            role not in {
                "user",
                "assistant",
            }
            or not content
        ):
            continue

        if (
            merged
            and merged[-1]["role"] == role
        ):

            merged[-1]["content"] += (
                "\n" + content
            )

        else:

            merged.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return merged


# ============================================================
# 6. MAKE VALID CONVERSATION
# ============================================================

def prepare_messages(messages):

    messages = merge_adjacent_messages(
        messages
    )

    # Remove anything before first USER.
    while (
        messages
        and messages[0]["role"] != "user"
    ):
        messages.pop(0)

    # We want the training conversation to end
    # with an assistant response.
    while (
        messages
        and messages[-1]["role"] != "assistant"
    ):
        messages.pop()

    if len(messages) < 2:
        return []

    has_user = any(
        m["role"] == "user"
        for m in messages
    )

    has_assistant = any(
        m["role"] == "assistant"
        for m in messages
    )

    if not (
        has_user
        and has_assistant
    ):
        return []

    return messages


# ============================================================
# 7. DEDUPLICATION HASH
# ============================================================

def conversation_hash(messages):

    parts = []

    for message in messages:

        text = message[
            "content"
        ].lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        parts.append(
            f"{message['role']}:{text}"
        )

    normalized = "\n".join(
        parts
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# 8. STANDARD VOYARI RECORD
# ============================================================

def build_record(
    source,
    messages,
    metadata=None,
):

    messages = prepare_messages(
        messages
    )

    if not messages:
        return None

    example_hash = conversation_hash(
        messages
    )

    return {
        "id": example_hash[:16],
        "source": source,
        "messages": messages,
        "metadata": metadata or {},
    }


# ============================================================
# 9. READ JSONL
# ============================================================

def read_jsonl(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                yield json.loads(
                    line
                )

            except json.JSONDecodeError:

                print(
                    f"WARNING: invalid JSON "
                    f"{path.name} "
                    f"line {line_number}"
                )


# ============================================================
# 10. INDIA TRAVEL ITINERARIES
# ============================================================

def convert_india_itineraries():

    for row in read_jsonl(
        INDIA_ITINERARIES_PATH
    ):

        instruction = clean_text(
            row.get(
                "instruction"
            )
        )

        user_input = clean_text(
            row.get(
                "input"
            )
        )

        answer = clean_text(
            row.get(
                "output"
            )
        )

        user_parts = []

        if instruction:
            user_parts.append(
                instruction
            )

        if user_input:
            user_parts.append(
                user_input
            )

        user_message = "\n\n".join(
            user_parts
        )

        record = build_record(
            source="india_travel_itineraries",
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ],
        )

        if record:
            yield record


# ============================================================
# 11. BITEXT TRAVEL
# ============================================================

def convert_bitext():

    for row in read_jsonl(
        BITEXT_PATH
    ):

        user_message = clean_text(
            row.get(
                "instruction"
            )
        )

        assistant_message = clean_text(
            row.get(
                "response"
            )
        )

        metadata = {
            "intent": row.get(
                "intent"
            ),
            "category": row.get(
                "category"
            ),
            "tags": row.get(
                "tags"
            ),
        }

        record = build_record(
            source="bitext_travel",
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": assistant_message,
                },
            ],
            metadata=metadata,
        )

        if record:
            yield record


# ============================================================
# 12. MULTIWOZ 2.2
# ============================================================

def convert_multiwoz():

    for row in read_jsonl(
        MULTIWOZ_PATH
    ):

        turns = row.get(
            "turns",
            {},
        )

        messages = []


        # ----------------------------------------------------
        # Hugging Face may store nested sequences as
        # dictionary-of-lists.
        # ----------------------------------------------------

        if isinstance(
            turns,
            dict,
        ):

            speakers = (
                turns.get("speaker")
                or []
            )

            utterances = (
                turns.get("utterance")
                or turns.get("text")
                or []
            )

            for speaker, utterance in zip(
                speakers,
                utterances,
            ):

                role = normalize_role(
                    speaker
                )

                if role:

                    messages.append(
                        {
                            "role": role,
                            "content": utterance,
                        }
                    )


        # ----------------------------------------------------
        # Or it may already be list-of-dictionaries.
        # ----------------------------------------------------

        elif isinstance(
            turns,
            list,
        ):

            for turn in turns:

                if not isinstance(
                    turn,
                    dict,
                ):
                    continue

                role = normalize_role(
                    turn.get(
                        "speaker"
                    )
                )

                utterance = (
                    turn.get(
                        "utterance"
                    )
                    or turn.get(
                        "text"
                    )
                )

                if role and utterance:

                    messages.append(
                        {
                            "role": role,
                            "content": utterance,
                        }
                    )


        record = build_record(
            source="multiwoz_2_2",
            messages=messages,
            metadata={
                "dialogue_id": row.get(
                    "dialogue_id"
                ),
                "services": row.get(
                    "services"
                ),
            },
        )

        if record:
            yield record


# ============================================================
# 13. TASKMASTER-2
# ============================================================

def convert_taskmaster():

    for row in read_jsonl(
        TASKMASTER_PATH
    ):

        messages = []

        utterances = row.get(
            "utterances",
            [],
        )

        for utterance in utterances:

            if not isinstance(
                utterance,
                dict,
            ):
                continue

            role = normalize_role(
                utterance.get(
                    "speaker"
                )
            )

            text = clean_text(
                utterance.get(
                    "text"
                )
            )

            if role and text:

                messages.append(
                    {
                        "role": role,
                        "content": text,
                    }
                )


        record = build_record(
            source="taskmaster_2",
            messages=messages,
            metadata={
                "conversation_id": row.get(
                    "conversation_id"
                ),
                "domain": row.get(
                    "voyari_domain"
                ),
            },
        )

        if record:
            yield record


# ============================================================
# 14. TRAVELPLANNER PLAN → READABLE TEXT
# ============================================================

def format_travelplanner_plan(
    annotated_plan
):

    day_records = []


    def collect_days(value):

        if isinstance(
            value,
            dict,
        ):

            if value.get(
                "days"
            ):

                day_records.append(
                    value
                )

        elif isinstance(
            value,
            list,
        ):

            for item in value:
                collect_days(
                    item
                )


    collect_days(
        annotated_plan
    )


    if not day_records:

        return clean_text(
            json.dumps(
                annotated_plan,
                ensure_ascii=False,
            )
        )


    result = []

    field_labels = [
        (
            "current_city",
            "Location",
        ),
        (
            "transportation",
            "Transportation",
        ),
        (
            "breakfast",
            "Breakfast",
        ),
        (
            "attraction",
            "Attractions",
        ),
        (
            "lunch",
            "Lunch",
        ),
        (
            "dinner",
            "Dinner",
        ),
        (
            "accommodation",
            "Accommodation",
        ),
    ]


    for day in day_records:

        day_number = day.get(
            "days"
        )

        result.append(
            f"Day {day_number}:"
        )

        for key, label in field_labels:

            value = clean_text(
                day.get(
                    key
                )
            )

            if (
                not value
                or value == "-"
            ):
                continue

            result.append(
                f"{label}: {value}"
            )

        result.append(
            ""
        )


    return "\n".join(
        result
    ).strip()


# ============================================================
# 15. TRAVELPLANNER
# ============================================================

def convert_travelplanner():

    for row in read_jsonl(
        TRAVELPLANNER_PATH
    ):

        query = clean_text(
            row.get(
                "query"
            )
        )

        plan = format_travelplanner_plan(
            row.get(
                "annotated_plan"
            )
        )


        metadata = {
            "origin": row.get(
                "org"
            ),
            "destination": row.get(
                "dest"
            ),
            "days": row.get(
                "days"
            ),
            "people": row.get(
                "people_number"
            ),
            "budget": row.get(
                "budget"
            ),
            "level": row.get(
                "level"
            ),

            # Important:
            # Do not treat old benchmark facts as
            # current factual travel information.
            "training_use":
                "planning_structure_only",

            "needs_fact_verification":
                True,
        }


        record = build_record(
            source="travelplanner",
            messages=[
                {
                    "role": "user",
                    "content": query,
                },
                {
                    "role": "assistant",
                    "content": plan,
                },
            ],
            metadata=metadata,
        )

        if record:
            yield record


# ============================================================
# 16. ALL NEW SFT SOURCES
# ============================================================

SOURCES = {
    "india_travel_itineraries":
        convert_india_itineraries,

    "multiwoz_2_2":
        convert_multiwoz,

    "bitext_travel":
        convert_bitext,

    "taskmaster_2":
        convert_taskmaster,

    "travelplanner":
        convert_travelplanner,
}


# ============================================================
# 17. WRITE NORMALIZED FILES
# ============================================================

def write_normalized_datasets():

    seen_hashes = set()

    total_saved = 0
    total_duplicates = 0


    print()
    print("=" * 70)
    print("VOYARILM SFT NORMALIZATION")
    print("=" * 70)


    for source_name, converter in SOURCES.items():

        output_path = (
            OUTPUT_ROOT
            / f"{source_name}.jsonl"
        )

        source_saved = 0
        source_duplicates = 0


        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for record in converter():

                example_hash = conversation_hash(
                    record["messages"]
                )

                if example_hash in seen_hashes:

                    source_duplicates += 1
                    total_duplicates += 1

                    continue

                seen_hashes.add(
                    example_hash
                )

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                source_saved += 1
                total_saved += 1


        print()
        print(
            f"{source_name}"
        )

        print(
            f"  saved      : "
            f"{source_saved:,}"
        )

        print(
            f"  duplicates : "
            f"{source_duplicates:,}"
        )

        print(
            f"  file       : "
            f"{output_path}"
        )


    print()
    print("-" * 70)

    print(
        f"TOTAL SAVED      : "
        f"{total_saved:,}"
    )

    print(
        f"TOTAL DUPLICATES : "
        f"{total_duplicates:,}"
    )

    print()
    print("NORMALIZED CANDIDATES:")
    print(OUTPUT_ROOT)

    print()
    print("=" * 70)


# ============================================================
# 18. RUN
# ============================================================

if __name__ == "__main__":

    write_normalized_datasets()
    