from src.data.pretraining_documents import (
    yield_pretraining_documents,
)

from src.data.pretraining_formatter import (
    format_pretraining_document,
)
from src.data.voyari_data_config import (
    CONTEXT_LENGTH,
)


# ==================================================
# 4. CREATE TRAINING SEQUENCES
# ==================================================

def yield_training_sequences():

    buffer = []

    for source, document in yield_pretraining_documents():

        token_ids = format_pretraining_document(
            document
        )

        buffer.extend(
            token_ids
        )

        # We need 1025 tokens:
        #
        # 1024 for x
        # plus
        # 1 extra token for y
        #
        while len(buffer) >= CONTEXT_LENGTH + 1:

            window = buffer[
                :CONTEXT_LENGTH + 1
            ]

            x = window[:-1]

            y = window[1:]

            yield x, y

            # Remove the 1024 tokens already used
            # as inputs.
            #
            # Keep the last token because it becomes
            # the first input token of the next window.
            buffer = buffer[
                CONTEXT_LENGTH:
            ]


# ==================================================
# 5. SMALL TEST
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("VOYARILM TRAINING SEQUENCE TEST")
    print("=" * 70)

    for sequence_number, (x, y) in enumerate(
        yield_training_sequences(),
        start=1,
    ):

        print()
        print(
            f"Sequence: {sequence_number}"
        )

        print(
            f"x length: {len(x)}"
        )

        print(
            f"y length: {len(y)}"
        )

        print()
        print("First 20 x token IDs:")

        print(
            x[:20]
        )

        print()
        print("First 20 y token IDs:")

        print(
            y[:20]
        )

        # Only inspect the first sequence.
        break