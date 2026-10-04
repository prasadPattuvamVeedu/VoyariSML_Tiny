import json
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
)

CANDIDATE_DIR = (
    DATA_ROOT
    / "04_INSTRUCTION"
    / "normalized_candidates"
)

FINAL_DIR = (
    DATA_ROOT
    / "06_FINAL_TRAINING"
    / "instruction"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. EXISTING FINAL INSTRUCTION DATASETS
# ============================================================

EXISTING_FINAL_FILES = [
    FINAL_DIR
    / "01_voyari_v9_train.jsonl",

    FINAL_DIR
    / "02_sgd_travel_clarification_train.jsonl",
]


# ============================================================
# 3. FIVE NEW NORMALIZED DATASETS
# ============================================================

NEW_DATASETS = {

    "india_travel_itineraries": (
        CANDIDATE_DIR
        / "india_travel_itineraries.jsonl",

        FINAL_DIR
        / "03_india_travel_itineraries_train.jsonl",
    ),

    "multiwoz_2_2": (
        CANDIDATE_DIR
        / "multiwoz_2_2.jsonl",

        FINAL_DIR
        / "04_multiwoz_2_2_train.jsonl",
    ),

    "bitext_travel": (
        CANDIDATE_DIR
        / "bitext_travel.jsonl",

        FINAL_DIR
        / "05_bitext_travel_train.jsonl",
    ),

    "taskmaster_2": (
        CANDIDATE_DIR
        / "taskmaster_2.jsonl",

        FINAL_DIR
        / "06_taskmaster_2_train.jsonl",
    ),

    "travelplanner": (
        CANDIDATE_DIR
        / "travelplanner.jsonl",

        FINAL_DIR
        / "07_travelplanner_train.jsonl",
    ),
}


# ============================================================
# 4. READ JSONL FILE
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

                record = json.loads(
                    line
                )

                yield record

            except json.JSONDecodeError:

                print(
                    f"Bad JSON skipped: "
                    f"{path.name} "
                    f"line {line_number}"
                )


# ============================================================
# 5. CONVERT CONVERSATION TO COMPARABLE TEXT
# ============================================================

def conversation_key(messages):

    parts = []

    for message in messages:

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            message.get(
                "content",
                "",
            )
        )

        # Example:
        #
        # "Plan   A Trip To Munnar"
        #
        # becomes:
        #
        # "plan a trip to munnar"

        content = " ".join(
            content.lower().split()
        )

        parts.append(
            f"{role}: {content}"
        )

    return "\n".join(
        parts
    )


# ============================================================
# 6. CHECK CONVERSATION STRUCTURE
# ============================================================

def valid_record(record):

    # Record must be a dictionary.
    if not isinstance(
        record,
        dict,
    ):

        return False


    messages = record.get(
        "messages"
    )


    # messages must be a list.
    if not isinstance(
        messages,
        list,
    ):

        return False


    # Minimum:
    #
    # user
    # assistant

    if len(messages) < 2:

        return False


    # Must start with user.
    if messages[0].get(
        "role"
    ) != "user":

        return False


    # Must finish with assistant.
    if messages[-1].get(
        "role"
    ) != "assistant":

        return False


    previous_role = None


    for message in messages:

        if not isinstance(
            message,
            dict,
        ):

            return False


        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )


        # Only user and assistant allowed.
        if role not in {
            "user",
            "assistant",
        }:

            return False


        # Content cannot be empty.
        if (
            not isinstance(
                content,
                str,
            )
            or not content.strip()
        ):

            return False


        # We do not want:
        #
        # user
        # user
        #
        # or
        #
        # assistant
        # assistant

        if role == previous_role:

            return False


        previous_role = role


    return True


# ============================================================
# 7. LOAD EXISTING V9 + SGD CONVERSATIONS
# ============================================================

def load_existing_conversations():

    seen_conversations = set()


    print()
    print(
        "Existing final instruction datasets"
    )

    print(
        "-" * 70
    )


    for path in EXISTING_FINAL_FILES:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing final dataset:\n"
                f"{path}"
            )


        count = 0


        for record in read_jsonl(
            path
        ):

            messages = record.get(
                "messages"
            )


            if not isinstance(
                messages,
                list,
            ):

                continue


            if not messages:

                continue


            key = conversation_key(
                messages
            )


            seen_conversations.add(
                key
            )


            count += 1


        print(
            f"{path.name}"
        )

        print(
            f"  conversations: "
            f"{count:,}"
        )


    return seen_conversations


# ============================================================
# 8. PROCESS ONE NEW DATASET
# ============================================================

def process_dataset(
    dataset_name,
    source_path,
    destination_path,
    seen_conversations,
):

    total = 0
    saved = 0
    duplicates = 0
    invalid = 0


    with destination_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:


        for record in read_jsonl(
            source_path
        ):

            total += 1


            # ------------------------------------------------
            # CHECK 1:
            # Is the conversation structurally valid?
            # ------------------------------------------------

            if not valid_record(
                record
            ):

                invalid += 1

                continue


            # ------------------------------------------------
            # CHECK 2:
            # TravelPlanner must contain readable Day output.
            # ------------------------------------------------

            if (
                dataset_name
                == "travelplanner"
            ):

                assistant_text = (
                    record[
                        "messages"
                    ][-1][
                        "content"
                    ]
                )


                if not assistant_text.lstrip().startswith(
                    "Day "
                ):

                    invalid += 1

                    continue


            # ------------------------------------------------
            # CHECK 3:
            # Is the same conversation already present?
            # ------------------------------------------------

            key = conversation_key(
                record[
                    "messages"
                ]
            )


            if key in seen_conversations:

                duplicates += 1

                continue


            # ------------------------------------------------
            # GOOD RECORD
            # ------------------------------------------------

            seen_conversations.add(
                key
            )


            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


            saved += 1


    # --------------------------------------------------------
    # PRINT RESULT FOR THIS DATASET
    # --------------------------------------------------------

    print()

    print(
        dataset_name
    )

    print(
        f"  total      : "
        f"{total:,}"
    )

    print(
        f"  saved      : "
        f"{saved:,}"
    )

    print(
        f"  duplicates : "
        f"{duplicates:,}"
    )

    print(
        f"  invalid    : "
        f"{invalid:,}"
    )

    print(
        f"  output     : "
        f"{destination_path.name}"
    )


    return (
        total,
        saved,
        duplicates,
        invalid,
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "VOYARILM FINAL INSTRUCTION DATASET"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # First remember conversations already inside:
    #
    # 01 Voyari V9
    # 02 SGD clarification
    # --------------------------------------------------------

    seen_conversations = (
        load_existing_conversations()
    )


    print()

    print(
        "Processing five new datasets"
    )

    print(
        "-" * 70
    )


    grand_total = 0
    grand_saved = 0
    grand_duplicates = 0
    grand_invalid = 0


    # --------------------------------------------------------
    # Process:
    #
    # India
    # MultiWOZ
    # Bitext
    # Taskmaster
    # TravelPlanner
    # --------------------------------------------------------

    for dataset_name, (
        source_path,
        destination_path,
    ) in NEW_DATASETS.items():


        if not source_path.exists():

            raise FileNotFoundError(
                f"Candidate dataset missing:\n"
                f"{source_path}"
            )


        (
            total,
            saved,
            duplicates,
            invalid,
        ) = process_dataset(
            dataset_name,
            source_path,
            destination_path,
            seen_conversations,
        )


        grand_total += total

        grand_saved += saved

        grand_duplicates += duplicates

        grand_invalid += invalid


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FINALIZATION RESULT"
    )

    print(
        "=" * 70
    )


    print(
        f"New records checked : "
        f"{grand_total:,}"
    )

    print(
        f"New records saved   : "
        f"{grand_saved:,}"
    )

    print(
        f"Duplicates removed  : "
        f"{grand_duplicates:,}"
    )

    print(
        f"Invalid removed     : "
        f"{grand_invalid:,}"
    )


    print()

    print(
        "Final instruction folder:"
    )

    print(
        FINAL_DIR
    )


    print()

    print(
        "Expected final files:"
    )

    print(
        "01_voyari_v9_train.jsonl"
    )

    print(
        "02_sgd_travel_clarification_train.jsonl"
    )

    print(
        "03_india_travel_itineraries_train.jsonl"
    )

    print(
        "04_multiwoz_2_2_train.jsonl"
    )

    print(
        "05_bitext_travel_train.jsonl"
    )

    print(
        "06_taskmaster_2_train.jsonl"
    )

    print(
        "07_travelplanner_train.jsonl"
    )


    print()

    print(
        "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":

    main()