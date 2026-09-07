import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.model_config import VOCAB_SIZE
from src.model.voyari_lm import VoyariLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


model = VoyariLM().to(device)

model.train()


# create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)

# For now these are example token IDs.
#
# Later these will come from your actual
# Voyari tokenizer + training dataset.
#
# Batch = 2
# Tokens per sequence = 5

token_ids = torch.tensor(
    [
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
    ],
    dtype=torch.long,
    device=device,
)

input_ids = token_ids[:, :-1]  # All tokens except the last one
labels = token_ids[:, 1:]  # All tokens except the first one
logits = model(input_ids)
print()
print("Logits minimum:", logits.min().item())
print("Logits maximum:", logits.max().item())
print("Logits mean   :", logits.mean().item())
print("Logits std    :", logits.std().item())
labels_flat = labels.reshape(-1)

print()
print("Input shape :", input_ids.shape)
print("Label shape :", labels.shape)
print("Logits shape:", logits.shape)

logits_flat = logits.view(-1, VOCAB_SIZE)
loss = F.cross_entropy(
    logits_flat,
    labels_flat,
)
loss = F.cross_entropy(
    logits_flat,
    labels_flat,
)


print()
print("Loss before update:", loss.item())


# ============================================================
# 10. Backward pass
# ============================================================

loss.backward()
torch.no_grad()  # Disable gradient tracking for inference
optimizer.step()


# loss.backward()
#
# calculates gradients for:
#
# TokenEmbedding
#
# Transformer Block 1
#   RMSNorm
#   Q projection
#   K projection
#   V projection
#   Attention output projection
#   MLP gate projection
#   MLP up projection
#   MLP down projection
#
# Transformer Block 2
# ...
#
# Transformer Block 14
#
# Final RMSNorm
#
# LM Head

with torch.no_grad():

    new_logits = model(input_ids)

    new_loss = F.cross_entropy(
        new_logits.reshape(-1, VOCAB_SIZE),
        labels_flat,
    )
print("Loss after update :", new_loss.item())


# ============================================================
# 14. Finish
# ============================================================

print()
print("One VoyariLM training step completed.")