import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import (
    D_MODEL,
    FFN_DIM,
    USE_BIAS
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, FFN_DIM, bias=USE_BIAS)
        self.up_proj = nn.Linear(D_MODEL, FFN_DIM, bias=USE_BIAS)
        self.down_proj = nn.Linear(FFN_DIM, D_MODEL, bias=USE_BIAS)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.silu(gate)
        up = self.up_proj(x)
        Output = self.down_proj(gate * up)
        return Output
# Attention decides which other tokens are relevant.
#  MLP/SwiGLU decides which learned features inside the 
# resulting token representation are useful 
# and how strongly to process them.