import json
import re
from pathlib import Path


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(r"D:\voyari_sml")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "01_RAW_SOURCES"
    / "alia_tourism"
    / "data"
    / "train"
    / "tourism-public-en-md.base.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "02_CLEAN_PRETRAINING"
    / "alia_tourism"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "alia_tourism_english_clean.txt"
)


# ==================================================
# 2. PROMOTIONAL LINES WE DO NOT WANT
# ==================================================

PROMOTIONAL_PHRASES = [
    "check availability",
    "check prices",
    "book now",
    "find the best hotel deals",
    "explore my shop",
    "amazon travel shop",
    "booking.com",
    "pre-filtered list of hotels",
    "top hotels in",
]


# ==================================================
# 3. CLEAN ONE DOCUMENT
# ==================================================

def clean_document(text):

    # ----------------------------------------------
    # Normalize line endings
    # ----------------------------------------------

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )


    # ----------------------------------------------
    # Remove Markdown images
    #
    # Example:
    #
    # ![Lisbon hotel](image.jpg)
    #
    # becomes:
    #
    # ""
    # ----------------------------------------------

    text = re.sub(
        r"!\[[^\]]*\]\([^)]*\)",
        "",
        text,
    )


    # ----------------------------------------------
    # Remove empty Markdown links left behind
    # after image removal.
    #
    # Example:
    #
    # [](https://example.com)
    #
    # becomes:
    #
    # ""
    # ----------------------------------------------

    text = re.sub(
        r"\[\s*\]\s*\([^)]*\)",
        "",
        text,
    )


    # ----------------------------------------------
    # Remove broken empty Markdown fragments
    #
    # Example:
    #
    # [](
    #
    # becomes:
    #
    # ""
    # ----------------------------------------------

    text = re.sub(
        r"\[\s*\]\s*\(",
        "",
        text,
    )


    # ----------------------------------------------
    # Convert normal Markdown links to plain text.
    #
    # Example:
    #
    # [Munnar](https://example.com)
    #
    # becomes:
    #
    # Munnar
    # ----------------------------------------------

    text = re.sub(
        r"\[([^\]]+)\]\([^)]*\)",
        r"\1",
        text,
    )


    # ----------------------------------------------
    # Remove any remaining raw URLs
    # ----------------------------------------------

    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )


    # ----------------------------------------------
    # Remove protocol-relative URLs
    #
    # Example:
    #
    # //www.example.com/image.jpg
    # ----------------------------------------------

    text = re.sub(
        r"//www\.\S+",
        "",
        text,
    )


    # ----------------------------------------------
    # Remove Markdown heading symbols
    #
    # Example:
    #
    # ### Best Places
    #
    # becomes:
    #
    # Best Places
    # ----------------------------------------------

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )


    # ----------------------------------------------
    # Remove bold Markdown markers
    #
    # **Munnar**
    #
    # becomes:
    #
    # Munnar
    # ----------------------------------------------

    text = text.replace(
        "**",
        "",
    )

    text = text.replace(
        "__",
        "",
    )


    # ----------------------------------------------
    # Remove escaped Markdown characters
    #
    # \*will\*
    #
    # becomes:
    #
    # will
    # ----------------------------------------------

    text = text.replace(
        r"\*",
        "",
    )

    text = text.replace(
        r"\_",
        "_",
    )


    # ----------------------------------------------
    # Remove remaining simple italic markers
    #
    # *fast*
    #
    # becomes:
    #
    # fast
    #
    # This does not remove normal star bullets yet.
    # ----------------------------------------------

    text = re.sub(
        r"\*([^*\n]+)\*",
        r"\1",
        text,
    )


    # ----------------------------------------------
    # Remove Markdown bullet symbols but KEEP text
    #
    # * Total trails: 83
    #
    # becomes:
    #
    # Total trails: 83
    # ----------------------------------------------

    text = re.sub(
        r"^\s*[\*\-]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )


    # ----------------------------------------------
    # Process document one line at a time
    # ----------------------------------------------

    cleaned_lines = []


    for line in text.splitlines():

        line = line.strip()


        # ------------------------------------------
        # Ignore empty lines inside document
        # ------------------------------------------

        if not line:
            continue


        # ------------------------------------------
        # Make lowercase copy ONLY for checking
        #
        # Original text is not lowercased.
        # ------------------------------------------

        lower_line = line.lower()


        # ------------------------------------------
        # Remove promotional / affiliate lines
        # ------------------------------------------

        if any(
            phrase in lower_line
            for phrase in PROMOTIONAL_PHRASES
        ):
            continue


        # ------------------------------------------
        # Remove any leftover broken Markdown-only
        # fragments.
        #
        # Examples:
        #
        # [](
        # []
        # ()
        # ------------------------------------------

        if line in {
            "[](",
            "[]",
            "()",
            "[",
            "]",
        }:
            continue


        # ------------------------------------------
        # Keep useful travel line
        # ------------------------------------------

        cleaned_lines.append(
            line
        )


    # ----------------------------------------------
    # Join useful lines back into one document
    # ----------------------------------------------

    cleaned_text = "\n".join(
        cleaned_lines
    )


    # ----------------------------------------------
    # Collapse repeated spaces/tabs
    #
    # "Munnar    Kerala"
    #
    # becomes:
    #
    # "Munnar Kerala"
    # ----------------------------------------------

    cleaned_text = re.sub(
        r"[ \t]+",
        " ",
        cleaned_text,
    )


    # ----------------------------------------------
    # Remove spaces before punctuation
    #
    # "Munnar ."
    #
    # becomes:
    #
    # "Munnar."
    # ----------------------------------------------

    cleaned_text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        cleaned_text,
    )


    # ----------------------------------------------
    # Reduce accidental repeated periods
    #
    # ".."
    #
    # becomes:
    #
    # "."
    # ----------------------------------------------

    cleaned_text = re.sub(
        r"\.{2,}",
        ".",
        cleaned_text,
    )


    return cleaned_text.strip()


# ==================================================
# 4. CLEAN COMPLETE DATASET
# ==================================================

def clean_dataset():

    # ----------------------------------------------
    # Create output directory if necessary
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ----------------------------------------------
    # Counters
    # ----------------------------------------------

    total_records = 0

    saved_documents = 0

    skipped_documents = 0


    # ----------------------------------------------
    # Open raw JSONL
    # ----------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as input_file:


        # ------------------------------------------
        # Open cleaned output file.
        #
        # "w" means overwrite previous cleaned file.
        # ------------------------------------------

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
        ) as output_file:


            # --------------------------------------
            # Read one JSON record at a time
            # --------------------------------------

            for line in input_file:

                line = line.strip()


                # ----------------------------------
                # Ignore blank JSONL lines
                # ----------------------------------

                if not line:
                    continue


                # ----------------------------------
                # Convert JSON string to dictionary
                # ----------------------------------

                record = json.loads(
                    line
                )


                total_records += 1


                # ----------------------------------
                # English only
                # ----------------------------------

                if record.get(
                    "language"
                ) != "en":

                    continue


                # ----------------------------------
                # Extract article text
                # ----------------------------------

                raw_text = record.get(
                    "text",
                    "",
                )


                # ----------------------------------
                # Clean article
                # ----------------------------------

                clean_text = clean_document(
                    raw_text
                )


                # ----------------------------------
                # Reject extremely small documents
                #
                # A tiny fragment is usually not
                # useful for pretraining.
                # ----------------------------------

                if len(clean_text) < 200:

                    skipped_documents += 1

                    continue


                # ----------------------------------
                # Save complete cleaned document
                # ----------------------------------

                output_file.write(
                    clean_text
                )


                # ----------------------------------
                # One blank line separates this
                # document from the next document.
                # ----------------------------------

                output_file.write(
                    "\n\n"
                )


                saved_documents += 1


    # ==================================================
    # 5. REPORT
    # ==================================================

    print()

    print("=" * 70)

    print(
        "ALIA TOURISM CLEANING COMPLETE"
    )

    print("=" * 70)


    print(
        f"Input records     : "
        f"{total_records:,}"
    )

    print(
        f"Saved documents   : "
        f"{saved_documents:,}"
    )

    print(
        f"Skipped documents : "
        f"{skipped_documents:,}"
    )


    print()

    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )


# ==================================================
# 6. RUN
# ==================================================

if __name__ == "__main__":

    clean_dataset()