import torch.nn as nn

from src.model.token_embedding import TokenEmbedding


class InputEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = TokenEmbedding()

    def forward(self, token_ids):

        embeddings = self.token_embedding(
            token_ids
        )

        return embeddings