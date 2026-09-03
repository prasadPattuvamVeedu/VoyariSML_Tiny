from pathlib import Path

from tokenizers import Tokenizer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(r"D:\voyari_sml")

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "voyari_tokenizer_16k_v2.json"
)

PRETRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "06_FINAL_TRAINING"
    / "pretraining"
)


# --------------------------------------------------
# 2. Approved pretraining files only
# --------------------------------------------------

PRETRAINING_FILES = [
    PRETRAINING_DIR / "01_wikivoyage_corpus.txt",
    PRETRAINING_DIR / "02_simplewiki_corpus.txt",
    PRETRAINING_DIR / "04_wikidata_india_travel_corpus.txt",
    PRETRAINING_DIR / "05_unesco_india_heritage_corpus.txt",
]


# --------------------------------------------------
# 3. Load the already-trained Voyari tokenizer
# --------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)


# --------------------------------------------------
# 4. Function to count tokens in one file
# --------------------------------------------------

def count_file_tokens(path):

    token_count = 0
    character_count = 0
    line_count = 0

    print()
    print(f"Counting: {path.name}")

    with open(
    path,
    "r",
    encoding="utf-8",
    errors="replace",
    ) as file:

        for line in file:

            line_count += 1

            text = line.strip()

            if not text:
                continue

            character_count += len(text)

            encoding = tokenizer.encode(text)

            token_count += len(encoding.ids)

    return (
        token_count,
        character_count,
        line_count,
    )


# --------------------------------------------------
# 5. Grand totals
# --------------------------------------------------

total_tokens = 0
total_characters = 0
total_lines = 0


# --------------------------------------------------
# 6. Count every approved corpus
# --------------------------------------------------

for path in PRETRAINING_FILES:

    tokens, characters, lines = count_file_tokens(
        path
    )

    total_tokens += tokens
    total_characters += characters
    total_lines += lines

    print(f"Characters  : {characters:,}")
    print(f"Lines       : {lines:,}")
    print(f"Tokens      : {tokens:,}")

    if tokens > 0:

        chars_per_token = (
            characters / tokens
        )

        print(
            f"Chars/token : {chars_per_token:.2f}"
        )


# --------------------------------------------------
# 7. Final result
# --------------------------------------------------

print()
print("=" * 70)

print("VOYARILM PRETRAINING TOKEN COUNT")

print("=" * 70)

print(
    f"Total characters : {total_characters:,}"
)

print(
    f"Total lines      : {total_lines:,}"
)

print(
    f"Total tokens     : {total_tokens:,}"
)


if total_tokens > 0:

    average_chars_per_token = (
        total_characters
        / total_tokens
    )

    print(
        "Average chars/token:",
        f"{average_chars_per_token:.2f}",
    )

print()
print("Finished.")