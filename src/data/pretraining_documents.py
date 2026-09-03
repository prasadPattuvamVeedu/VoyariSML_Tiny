from pathlib import Path


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(r"D:\voyari_sml")

PRETRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "06_FINAL_TRAINING"
    / "pretraining"
)


# ==================================================
# 2. FINAL APPROVED PRETRAINING FILES
# ==================================================

WIKIVOYAGE_FILE = (
    PRETRAINING_DIR
    / "01_wikivoyage_corpus.txt"
)

SIMPLEWIKI_FILE = (
    PRETRAINING_DIR
    / "02_simplewiki_corpus.txt"
)

WIKIDATA_FILE = (
    PRETRAINING_DIR
    / "04_wikidata_india_travel_corpus.txt"
)

UNESCO_FILE = (
    PRETRAINING_DIR
    / "05_unesco_india_heritage_corpus.txt"
)


# ==================================================
# 3. EXPECTED DOCUMENT COUNTS
# ==================================================

EXPECTED_COUNTS = {
    "wikivoyage": 30_036,
    "simplewiki": 133_230,
    "wikidata_india_travel": 2_197,
    "unesco_india_heritage": 45,
}


# ==================================================
# 4. CHECK THAT FILE EXISTS
# ==================================================

def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )


# ==================================================
# 5. READ COMPLETE DOCUMENTS
# ==================================================
#
# These four corpora use blank lines to separate
# complete documents.
#
# Example:
#
# Munnar line 1
# Munnar line 2
#
# Kochi line 1
# Kochi line 2
#
# becomes:
#
# DOCUMENT 1:
# Munnar line 1
# Munnar line 2
#
# DOCUMENT 2:
# Kochi line 1
# Kochi line 2
#
# ==================================================

def yield_blank_separated_documents(path):

    check_file(path)

    document_lines = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line in file:

            text = line.strip()

            # --------------------------------------
            # Blank line = current document finished
            # --------------------------------------

            if not text:

                if document_lines:

                    document = "\n".join(
                        document_lines
                    )

                    yield document

                    # Empty the basket for the
                    # next document.
                    document_lines = []

                continue

            # --------------------------------------
            # Normal line
            # --------------------------------------

            document_lines.append(
                text
            )

    # ----------------------------------------------
    # If the file ends without a final blank line,
    # return the last document too.
    # ----------------------------------------------

    if document_lines:

        document = "\n".join(
            document_lines
        )

        yield document


# ==================================================
# 6. MASTER PRETRAINING DOCUMENT GENERATOR
# ==================================================
#
# Gives future training code:
#
# (
#     source_name,
#     complete_document_text
# )
#
# one document at a time.
# ==================================================

def yield_pretraining_documents():

    # Wikivoyage
    for document in yield_blank_separated_documents(
        WIKIVOYAGE_FILE
    ):

        yield (
            "wikivoyage",
            document,
        )


    # Simple English Wikipedia
    for document in yield_blank_separated_documents(
        SIMPLEWIKI_FILE
    ):

        yield (
            "simplewiki",
            document,
        )


    # Wikidata India Travel
    for document in yield_blank_separated_documents(
        WIKIDATA_FILE
    ):

        yield (
            "wikidata_india_travel",
            document,
        )


    # UNESCO India Heritage
    for document in yield_blank_separated_documents(
        UNESCO_FILE
    ):

        yield (
            "unesco_india_heritage",
            document,
        )


# ==================================================
# 7. TEST DOCUMENT BOUNDARIES
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("VOYARILM V2 PRETRAINING DOCUMENT TEST")
    print("=" * 70)

    counts = {}

    first_examples = {}


    # Read every complete document.
    for source, document in yield_pretraining_documents():

        counts[source] = (
            counts.get(source, 0)
            + 1
        )

        # Save only the first document from
        # each source for visual checking.
        if source not in first_examples:

            first_examples[source] = document


    print()
    print("DOCUMENT COUNTS")
    print("-" * 70)

    total_documents = 0

    for source, expected in EXPECTED_COUNTS.items():

        actual = counts.get(
            source,
            0,
        )

        total_documents += actual

        if actual == expected:
            status = "PASS"
        else:
            status = "CHECK"

        print(
            f"{source:28} "
            f"actual={actual:>8,} "
            f"expected={expected:>8,} "
            f"{status}"
        )


    expected_total = sum(
        EXPECTED_COUNTS.values()
    )

    print()
    print(
        f"Actual total documents   : "
        f"{total_documents:,}"
    )

    print(
        f"Expected total documents : "
        f"{expected_total:,}"
    )


    print()
    print("=" * 70)
    print("FIRST DOCUMENT FROM EACH SOURCE")
    print("=" * 70)


    for source in EXPECTED_COUNTS:

        document = first_examples.get(
            source
        )

        print()
        print(f"SOURCE: {source}")
        print("-" * 70)

        if document is None:

            print("No document returned.")

        else:

            # Only show first 500 characters.
            print(
                document[:500]
            )


    print()
    print("=" * 70)
    print("Finished.")