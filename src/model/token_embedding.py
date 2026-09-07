import torch
import torch.nn as nn

from configs.model_config import (
    VOCAB_SIZE,
    D_MODEL,
)


class TokenEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=D_MODEL,
        )

    def forward(self, token_ids):

        embeddings = self.embedding(
            token_ids
        )

        return embeddings