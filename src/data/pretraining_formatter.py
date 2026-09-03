from pathlib import Path
from tokenizers import Tokenizer



PROJECT_ROOT = Path(r"D:\voyari_sml")

Tokenizer_path = TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "voyari_tokenizer_16k_v2.json"
)

tokenizer  = Tokenizer.from_file(str(TOKENIZER_PATH))

BOS_ID = tokenizer.token_to_id("<bos>")
EOS_ID = tokenizer.token_to_id("<eos>")

# Make sure the tokenizer really contains them
if BOS_ID is None:
    raise ValueError(
        "<bos> token was not found in the tokenizer."
    )

if EOS_ID is None:
    raise ValueError(
        "<eos> token was not found in the tokenizer."
    )

# --------------------------------------------------
# 4. Format ONE complete pretraining document
# --------------------------------------------------

def format_pretraining_document(text):

    # Remove unnecessary whitespace only from
    # the beginning and end of the document.
    text = text.strip()

    # Ignore empty documents.
    if not text:
        return []

    # Convert document text into normal token IDs.
    encoding = tokenizer.encode(text)

    document_token_ids = encoding.ids

    # Add:
    #
    # <bos> document text <eos>
    #
    # For Voyari tokenizer:
    #
    # <bos> = 1
    # <eos> = 2
    #
    formatted_token_ids = (
        [BOS_ID]
        + document_token_ids
        + [EOS_ID]
    )

    return formatted_token_ids


# --------------------------------------------------
# 5. Test only when this file is run directly
# --------------------------------------------------

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

    # [1:-1] removes BOS and EOS only for this
    # decoding test.
    decoded_text = tokenizer.decode(
        formatted_ids[1:-1]
    )

    print(decoded_text)

    print()
    print("Finished.")