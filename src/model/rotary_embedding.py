import torch
import torch.nn as nn

from configs.model_config import HEAD_DIM, CONTEXT_LENGTH


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=HEAD_DIM,
        max_seq_len=CONTEXT_LENGTH,
        base=10000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(self, q, k):
        # q, k expected shape:
        # [batch, NUM_HEADS, sequence_length, HEAD_DIM]
        #
        # With your config:
        # [2, 6, 1024, 64]

        seq_len = q.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"CONTEXT_LENGTH={self.max_seq_len}"
            )

        positions = torch.arange(
            seq_len,
            device=q.device,
            dtype=self.inv_freq.dtype,
        )

        frequencies = torch.outer(
            positions,
            self.inv_freq,
        )

        cos = frequencies.cos()
        sin = frequencies.sin()

        # [sequence, 32]
        # becomes [1, 1, sequence, 32]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)

        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        q_rotated = torch.stack(
            (
                q_even * cos - q_odd * sin,
                q_even * sin + q_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)

        k_rotated = torch.stack(
            (
                k_even * cos - k_odd * sin,
                k_even * sin + k_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)

        return q_rotated, k_rotated