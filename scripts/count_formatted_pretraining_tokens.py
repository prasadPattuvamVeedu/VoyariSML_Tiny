from pathlib import Path
import sys


# ==================================================
# 1. PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(r"D:\voyari_sml")

# Allow Python to import files from the project.
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ==================================================
# 2. IMPORT OUR EXISTING PIPELINE
# ==================================================

from src.data.pretraining_documents import (
    yield_pretraining_documents,
)

from src.data.pretraining_formatter import (
    format_pretraining_document,
)


# ==================================================
# 3. COUNT EXACT FORMATTED TOKENS
# ==================================================

def count_formatted_tokens():

    total_documents = 0
    total_tokens = 0

    source_documents = {}
    source_tokens = {}

    print()
    print("=" * 70)
    print("VOYARILM V2 EXACT FORMATTED TOKEN COUNT")
    print("=" * 70)


    for source, document in yield_pretraining_documents():

        # ------------------------------------------
        # Convert ONE complete document:
        #
        # text
        #   ↓
        # tokenizer
        #   ↓
        # [BOS, token IDs..., EOS]
        # ------------------------------------------

        token_ids = format_pretraining_document(
            document
        )

        document_token_count = len(
            token_ids
        )


        # ------------------------------------------
        # Global totals
        # ------------------------------------------

        total_documents += 1

        total_tokens += (
            document_token_count
        )


        # ------------------------------------------
        # Per-source document count
        # ------------------------------------------

        source_documents[source] = (
            source_documents.get(source, 0)
            + 1
        )


        # ------------------------------------------
        # Per-source token count
        # ------------------------------------------

        source_tokens[source] = (
            source_tokens.get(source, 0)
            + document_token_count
        )


        # ------------------------------------------
        # Progress message
        # ------------------------------------------

        if total_documents % 10_000 == 0:

            print(
                f"Processed "
                f"{total_documents:,} documents..."
            )


    # ==================================================
    # 4. SHOW RESULTS
    # ==================================================

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)


    for source in source_documents:

        print()

        print(
            f"{source}"
        )

        print(
            f"  Documents : "
            f"{source_documents[source]:,}"
        )

        print(
            f"  Tokens    : "
            f"{source_tokens[source]:,}"
        )


    print()
    print("=" * 70)

    print(
        f"Total documents : "
        f"{total_documents:,}"
    )

    print(
        f"Total formatted tokens : "
        f"{total_tokens:,}"
    )

    print("=" * 70)

    print()
    print("Finished.")


# ==================================================
# 5. RUN ONLY WHEN THIS FILE IS EXECUTED DIRECTLY
# ==================================================

if __name__ == "__main__":

    count_formatted_tokens()