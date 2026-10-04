import ast
import json
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "04_INSTRUCTION"
    / "normalized_candidates"
    / "travelplanner.jsonl"
)

FINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "06_FINAL_TRAINING"
    / "instruction"
    / "07_travelplanner_train.jsonl"
)


# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    return " ".join(
        str(value).split()
    ).strip()


# ============================================================
# 3. REMOVE NESTED STRING LAYERS
# ============================================================

def parse_nested_value(value):

    # Maximum 6 layers just as a safety limit.
    for _ in range(6):

        # If it is already a list/dict,
        # parsing is finished.
        if not isinstance(
            value,
            str,
        ):
            return value

        text = value.strip()

        if not text:
            return None


        # --------------------------------------------
        # First try JSON parsing.
        #
        # Example:
        #
        # "\"[{'days': 1}]\""
        #
        # may become:
        #
        # "[{'days': 1}]"
        # --------------------------------------------

        try:

            value = json.loads(
                text
            )

            continue

        except json.JSONDecodeError:

            pass


        # --------------------------------------------
        # If it is Python-style text:
        #
        # "[{'days': 1}]"
        #
        # convert it using literal_eval().
        # --------------------------------------------

        try:

            value = ast.literal_eval(
                text
            )

            continue

        except (
            ValueError,
            SyntaxError,
        ):

            return None


    return value


# ============================================================
# 4. FIND ACTUAL DAY RECORDS
# ============================================================

def find_days(value):

    value = parse_nested_value(
        value
    )

    days = []


    # --------------------------------------------
    # DICTIONARY
    # --------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        # Important:
        #
        # TravelPlanner's trip-info dictionary also
        # contains "days": 3.
        #
        # So "days" alone is NOT enough.
        #
        # A real day record also contains fields such
        # as current_city, breakfast, attraction, etc.
        # --------------------------------------------

        day_fields = {
            "current_city",
            "transportation",
            "breakfast",
            "attraction",
            "lunch",
            "dinner",
            "accommodation",
        }

        is_real_day = (
            "days" in value
            and any(
                field in value
                for field in day_fields
            )
        )


        if is_real_day:

            days.append(
                value
            )


        # Search deeper inside nested structures.
        for child in value.values():

            if isinstance(
                child,
                (
                    dict,
                    list,
                    tuple,
                    str,
                ),
            ):

                days.extend(
                    find_days(
                        child
                    )
                )


    # --------------------------------------------
    # LIST OR TUPLE
    # --------------------------------------------

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        for child in value:

            days.extend(
                find_days(
                    child
                )
            )


    return days


# ============================================================
# 5. FORMAT TRAVEL PLAN
# ============================================================

def format_plan(raw_plan):

    # Find Day 1, Day 2, Day 3...
    day_records = find_days(
        raw_plan
    )


    if not day_records:

        return None


    lines = []


    fields = [
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


        lines.append(
            f"Day {day_number}:"
        )


        for key, label in fields:

            value = clean_text(
                day.get(
                    key
                )
            )


            # TravelPlanner uses "-"
            # when information is missing.
            if (
                not value
                or value == "-"
            ):

                continue


            lines.append(
                f"{label}: {value}"
            )


        # Empty line between days.
        lines.append(
            ""
        )


    return "\n".join(
        lines
    ).strip()


# ============================================================
# 6. MAIN
# ============================================================

def main():

    saved = 0
    failed = 0


    with CANDIDATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as source_file, FINAL_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_file:


        for number, line in enumerate(
            source_file,
            start=1,
        ):

            line = line.strip()


            if not line:

                continue


            # ----------------------------------------
            # JSON text -> Python dictionary
            # ----------------------------------------

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                print(
                    f"Record {number}: "
                    f"bad JSON"
                )

                failed += 1

                continue


            # ----------------------------------------
            # GET MESSAGES
            # ----------------------------------------

            messages = record.get(
                "messages",
                [],
            )


            if len(messages) < 2:

                print(
                    f"Record {number}: "
                    f"missing messages"
                )

                failed += 1

                continue


            # ----------------------------------------
            # USER QUESTION
            # ----------------------------------------

            user_text = clean_text(
                messages[0].get(
                    "content"
                )
            )


            # ----------------------------------------
            # CURRENT BROKEN ASSISTANT PLAN
            # ----------------------------------------

            raw_plan = (
                messages[-1].get(
                    "content",
                    "",
                )
            )


            # ----------------------------------------
            # FIX PLAN
            # ----------------------------------------

            fixed_plan = format_plan(
                raw_plan
            )


            if not user_text:

                print(
                    f"Record {number}: "
                    f"empty user text"
                )

                failed += 1

                continue


            if not fixed_plan:

                print(
                    f"Record {number}: "
                    f"could not find day records"
                )

                failed += 1

                continue


            # ----------------------------------------
            # PRESERVE USEFUL METADATA
            # ----------------------------------------

            metadata = record.get(
                "metadata"
            )


            if not isinstance(
                metadata,
                dict,
            ):

                metadata = {}


            metadata[
                "training_use"
            ] = "planning_structure_only"


            metadata[
                "needs_fact_verification"
            ] = True


            # ----------------------------------------
            # BUILD FINAL TRAINING RECORD
            # ----------------------------------------

            final_record = {

                "id":
                    f"travelplanner_{number:05d}",

                "source":
                    "travelplanner",

                "messages": [
                    {
                        "role":
                            "user",

                        "content":
                            user_text,
                    },
                    {
                        "role":
                            "assistant",

                        "content":
                            fixed_plan,
                    },
                ],

                "metadata":
                    metadata,
            }


            # ----------------------------------------
            # PYTHON DICTIONARY -> JSON STRING
            # ----------------------------------------

            output_file.write(

                json.dumps(
                    final_record,
                    ensure_ascii=False,
                )

                + "\n"
            )


            saved += 1


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 60)

    print(
        "TRAVELPLANNER FINALIZATION"
    )

    print("=" * 60)

    print(
        f"Saved  : {saved}"
    )

    print(
        f"Failed : {failed}"
    )

    print()

    print(
        f"Output : {FINAL_FILE}"
    )

    print("=" * 60)


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":

    main()