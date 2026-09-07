import torch
import torch.nn as nn

from configs.model_config import (
    VOCAB_SIZE,
    D_MODEL,
    NUM_LAYERS,
    TIE_EMBEDDINGS,
    USE_BIAS,
)

from src.model.input_embedding import InputEmbedding
from src.model.transformer_block import TransformerBlock
from src.model.rms_norm import RMSNorm


class VoyariLM(nn.Module):

    def __init__(self):
        super().__init__()

        # ====================================================
        # 1. Input embedding
        # ====================================================

        self.input_embedding = InputEmbedding()


        # ====================================================
        # 2. Transformer blocks
        # ====================================================

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )


        # ====================================================
        # 3. Final normalization
        # ====================================================

        self.final_norm = RMSNorm()


        # ====================================================
        # 4. Language model head
        # ====================================================

        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=USE_BIAS,
        )


        # ====================================================
        # 5. Initialize weights
        # ====================================================

        self.apply(
            self._init_weights
        )


        # ====================================================
        # 6. Tie TokenEmbedding and LM Head
        # ====================================================

        if TIE_EMBEDDINGS:

            self.lm_head.weight = (
                self.input_embedding
                .token_embedding
                .embedding
                .weight
            )


    # ========================================================
    # Weight initialization
    # ========================================================

    def _init_weights(self, module):

        # Linear layers:
        #
        # Q
        # K
        # V
        # attention output
        # MLP
        # LM head

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


        # Embedding layers:
        #
        # TokenEmbedding
        # PositionalEmbedding

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, token_ids):

        # ----------------------------------------------------
        # Token IDs
        #
        # [B, T]
        # ----------------------------------------------------

        x = self.input_embedding(
            token_ids
        )


        # ----------------------------------------------------
        # Embeddings
        #
        # [B, T, 384]
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(x)


        # ----------------------------------------------------
        # 384 features -> 16,000 vocabulary scores
        # ----------------------------------------------------

        logits = self.lm_head(x)


        # Shape:
        #
        # [B, T, 16000]

        return logits