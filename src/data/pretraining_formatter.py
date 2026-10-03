from tokenizers import Tokenizer

from src.data.voyari_data_config import TOKENIZER_PATH


# ============================================================
# 1. LOAD THE VOYARILM TOKENIZER
# ============================================================

if not TOKENIZER_PATH.is_file():

    raise FileNotFoundError(
        f"Tokenizer file was not found:\n{TOKENIZER_PATH}"
    )


tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)


# ============================================================
# 2. REQUIRED PRETRAINING BOUNDARY TOKENS
# ============================================================

BOS_ID = tokenizer.token_to_id("<bos>")
EOS_ID = tokenizer.token_to_id("<eos>")


if BOS_ID is None:

    raise ValueError(
        "<bos> token was not found in the tokenizer."
    )


if EOS_ID is None:

    raise ValueError(
        "<eos> token was not found in the tokenizer."
    )


# ============================================================
# 3. FORMAT ONE COMPLETE PRETRAINING DOCUMENT
# ============================================================

def format_pretraining_document(text):
    """
    Convert one complete pretraining document into token IDs.

    Output:
        <bos> document tokens <eos>

    BOS and EOS are added once per document, not once per
    sentence and not once per 1024-token training window.
    """

    text = text.strip()

    if not text:

        return []

    # We add BOS/EOS ourselves below.
    # Keeping add_special_tokens=False prevents accidental
    # duplicate boundary tokens if the tokenizer later gets
    # a post-processor that inserts special tokens.
    encoding = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    document_token_ids = encoding.ids

    return (
        [BOS_ID]
        + document_token_ids
        + [EOS_ID]
    )


# ============================================================
# 4. SMALL STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    sample_document = (
        "Munnar is a hill station in Kerala. "
        "It is known for tea plantations. "
        "Many travellers visit the region."
    )

    formatted_ids = format_pretraining_document(
        sample_document
    )

    print()
    print("=" * 70)
    print("VOYARILM PRETRAINING FORMATTER TEST")
    print("=" * 70)

    print()
    print("Tokenizer:")
    print(TOKENIZER_PATH)

    print()
    print("Original document:")
    print(sample_document)

    print()
    print("Boundary token IDs:")
    print("<bos>", "->", BOS_ID)
    print("<eos>", "->", EOS_ID)

    print()
    print("Formatted token IDs:")
    print(formatted_ids)

    print()
    print("First token is BOS:")
    print(formatted_ids[0] == BOS_ID)

    print()
    print("Last token is EOS:")
    print(formatted_ids[-1] == EOS_ID)

    print()
    print("Total formatted tokens:")
    print(len(formatted_ids))

    print()
    print("Decoded document without BOS/EOS:")

    decoded_text = tokenizer.decode(
        formatted_ids[1:-1]
    )

    print(decoded_text)

    print()
    print("Finished.")
