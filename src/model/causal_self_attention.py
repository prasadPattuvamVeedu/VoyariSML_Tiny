import torch
import torch.nn as nn
import torch.nn.functional as F


from configs.model_config import (
    D_MODEL,
    NUM_HEADS,
    HEAD_DIM,
    USE_BIAS,
)

from src.model.rotary_embedding import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM

        self.q_proj = nn.Linear(
            D_MODEL,
            D_MODEL,
            bias=USE_BIAS,
        )

        self.k_proj = nn.Linear(
            D_MODEL,
            D_MODEL,
            bias=USE_BIAS,
        )

        self.v_proj = nn.Linear(
            D_MODEL,
            D_MODEL,
            bias=USE_BIAS,
        )

        self.out_proj = nn.Linear(
            D_MODEL,
            D_MODEL,
            bias=USE_BIAS,
        )

        self.rope = RotaryEmbedding()

    def forward(self,x):
        B,T,C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B,T,self.num_heads,self.head_dim)  
        k = k.view(B,T,self.num_heads,self.head_dim)  
        v = v.view(B,T,self.num_heads,self.head_dim) 

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) 

           # ----------------------------------
        # 4. Apply Rotary Embedding
        # ----------------------------------

        q, k = self.rope(q, k)

        # q shape remains:
        # [B, NUM_HEADS, T, HEAD_DIM]
        #
        # k shape remains:
        # [B, NUM_HEADS, T, HEAD_DIM]
        #
        # V is NOT rotated.


        # ----------------------------------
        # 5. Causal Self-Attention
        # ----------------------------------

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,)

         # ----------------------------------
        # 6. Put tokens before heads again
        # ----------------------------------

        attention_output = attention_output.transpose(1, 2)

        # [B, NUM_HEADS, T, HEAD_DIM]
        #
        # becomes
        #
        # [B, T, NUM_HEADS, HEAD_DIM]
        #
        # Example:
        # [2, 1024, 6, 64]


        # ----------------------------------
        # 7. Merge all heads
        # ----------------------------------

        attention_output = attention_output.contiguous().view(
            B,
            T,
            C,
        )

        # 6 heads × 64
        #       =
        # 384
        #
        # Shape becomes:
        # [B, T, D_MODEL]
        #
        # Example:
        # [2, 1024, 384]


        # ----------------------------------
        # 8. Final output projection
        # ----------------------------------

        output = self.out_proj(attention_output)

        # Shape:
        # [B, T, D_MODEL]

        return output