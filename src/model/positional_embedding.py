import torch
import torch.nn as nn

from configs.model_config import (
    CONTEXT_LENGTH,
    D_MODEL,
)


class PositionalEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=CONTEXT_LENGTH,
            embedding_dim=D_MODEL,
        )

    def forward(self, token_ids):

        batch_size, sequence_length = (
            token_ids.shape
        )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        position_embeddings = self.embedding(
            positions
        )

        return position_embeddings