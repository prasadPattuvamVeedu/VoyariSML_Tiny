import torch

from torch.utils.data import (
    IterableDataset,
    DataLoader,
)

from src.data.pretraining_sequences import (
    yield_training_sequences,
)


class PretrainingDataset(IterableDataset):

    def __iter__(self):

        for x, y in yield_training_sequences():

            x_tensor = torch.tensor(
                x,
                dtype=torch.long,
            )

            y_tensor = torch.tensor(
                y,
                dtype=torch.long,
            )

            yield x_tensor, y_tensor


def create_pretraining_dataloader(
    batch_size=1,
):

    dataset = PretrainingDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return dataloader


# ============================================================
# Small test
# ============================================================

if __name__ == "__main__":

    dataloader = create_pretraining_dataloader(
        batch_size=1
    )

    for x, y in dataloader:

        print()
        print("x shape:", x.shape)
        print("y shape:", y.shape)

        print()
        print("First 20 x IDs:")
        print(x[0, :20])

        print()
        print("First 20 y IDs:")
        print(y[0, :20])

        break