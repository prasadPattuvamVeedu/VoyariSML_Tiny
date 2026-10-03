from tokenizers import Tokenizer

from src.data.voyari_data_config import TOKENIZER_PATH


if not TOKENIZER_PATH.exists():

    raise FileNotFoundError(
        f"Tokenizer file was not found:\n{TOKENIZER_PATH}"
    )


tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

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


# --------------------------------------------------
# Format ONE complete pretraining document
# --------------------------------------------------

def format_pretraining_document(text):

    text = text.strip()

    if not text:

        return []

    encoding = tokenizer.encode(text)

    document_token_ids = encoding.ids

    formatted_token_ids = (
        [BOS_ID]
        + document_token_ids
        + [EOS_ID]
    )

    return formatted_token_ids


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
    print("Original document:")
    print(sample_document)

    print()
    print("BOS token:")
    print("<bos>", "->", BOS_ID)

    print()
    print("EOS token:")
    print("<eos>", "->", EOS_ID)

    print()
    print("Formatted token IDs:")
    print(formatted_ids)

    print()
    print("First token ID:")
    print(formatted_ids[0])

    print()
    print("Last token ID:")
    print(formatted_ids[-1])

    print()
    print("Total formatted tokens:")
    print(len(formatted_ids))

    print()
    print("Document text decoded again:")

    decoded_text = tokenizer.decode(
        formatted_ids[1:-1]
    )

    print(decoded_text)

    print()
    print("Finished.")
