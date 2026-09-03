import torch
import torch.nn as nn

from config.model_config import (vocab_size, D_model)

class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_model)

        def forward(self,token_ids):
            embedding = self.embedding(token_ids)
            return embedding

