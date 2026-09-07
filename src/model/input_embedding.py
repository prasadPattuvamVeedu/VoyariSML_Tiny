import torch.nn as nn

from src.model.token_embedding import (
    TokenEmbedding,
)

from src.model.positional_embedding import (
    PositionalEmbedding,
)


class InputEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = (
            TokenEmbedding()
        )

        self.positional_embedding = (
            PositionalEmbedding()
        )

    def forward(self, token_ids):

        token_embeddings = (
            self.token_embedding(
                token_ids
            )
        )

        position_embeddings = (
            self.positional_embedding(
                token_ids
            )
        )

        embeddings = (
            token_embeddings
            + position_embeddings
        )

        return embeddings