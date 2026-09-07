import torch
import torch.nn as nn

from src.model.rms_norm import RMSNorm
from src.model.causal_self_attention import CausalSelfAttention
from src.model.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm()
        self.attn = CausalSelfAttention()
        self.mlp_norm = RMSNorm()
        self.mlp = MLP()

    def forward(self, x):
        norm_in = self.norm(x)
        attn_out = self.attn(norm_in)
        x = x + attn_out
        mlp_norm_in=self.mlp_norm(x)
        mlp_out = self.mlp(mlp_norm_in)
        x = x + mlp_out 
        return x