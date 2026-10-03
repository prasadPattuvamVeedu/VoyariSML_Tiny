from src.data.voyari_data_config import PRETRAINING_SOURCES


# ==================================================
# 1. EXPECTED DOCUMENT COUNTS
# ==================================================

EXPECTED_COUNTS = {
    "wikivoyage": 30_036,
    "simplewiki": 133_230,
    "wikidata_india_travel": 2_197,
    "unesco_india_heritage": 45,
}


# ==================================================
# 2. CHECK THAT FILE EXISTS
# ==================================================

def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file was not found:\n{path}\n"
            "If this repository was cloned from GitHub, run "
            "`git lfs pull` so the training corpora are downloaded."
        )


# ==================================================
# 3. READ COMPLETE DOCUMENTS
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

            if not text:

                if document_lines:

                    yield "\n".join(
                        document_lines
                    )

                    document_lines = []

                continue

            document_lines.append(
                text
            )

    if document_lines:

        yield "\n".join(
            document_lines
        )


# ==================================================
# 4. MASTER PRETRAINING DOCUMENT GENERATOR
# ==================================================

def yield_pretraining_documents():

    for source, path in PRETRAINING_SOURCES.items():

        for document in yield_blank_separated_documents(
            path
        ):

            yield (
                source,
                document,
            )


# ==================================================
# 5. TEST DOCUMENT BOUNDARIES
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("VOYARILM PRETRAINING DOCUMENT TEST")
    print("=" * 70)

    counts = {}
    first_examples = {}

    for source, document in yield_pretraining_documents():

        counts[source] = (
            counts.get(source, 0)
            + 1
        )

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

        status = (
            "PASS"
            if actual == expected
            else "CHECK"
        )

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

            print(
                document[:500]
            )

    print()
    print("=" * 70)
    print("Finished.")
