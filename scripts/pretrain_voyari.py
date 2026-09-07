from pathlib import Path

import torch
import torch.nn.functional as F

from configs.model_config import VOCAB_SIZE

from src.model.voyari_lm import VoyariLM

from src.data.pretraining_dataset import (
    create_pretraining_dataloader,
)


# ============================================================
# 1. TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 1

LEARNING_RATE = 3e-4

# For resume test:
#
# checkpoint already contains step 5
#
# so we will continue until step 6.
MAX_STEPS = 6

LOG_EVERY = 1

SAVE_EVERY = 5


# ============================================================
# 2. CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR = Path(
    "artifacts/checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. CHECKPOINT TO RESUME FROM
# ============================================================

RESUME_CHECKPOINT = (
    CHECKPOINT_DIR
    / "voyari_step_5.pt"
)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Using device:", device)


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = VoyariLM().to(
    device
)


# ============================================================
# 6. CREATE OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 7. LOAD CHECKPOINT
# ============================================================

start_step = 0

if RESUME_CHECKPOINT.exists():

    print()
    print(
        "Loading checkpoint:",
        RESUME_CHECKPOINT,
    )

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=device,
    )

    # --------------------------------------------------------
    # Restore VoyariLM learned weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Restore AdamW optimizer state
    # --------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Restore training step
    # --------------------------------------------------------

    start_step = checkpoint[
        "step"
    ]

    print(
        "Resuming from step:",
        start_step,
    )

    print(
        "Previous loss:",
        checkpoint["loss"],
    )

else:

    print()
    print(
        "No resume checkpoint found."
    )

    print(
        "Training will start from step 0."
    )


# ============================================================
# 8. TRAINING MODE
# ============================================================

model.train()


# ============================================================
# 9. CREATE REAL VOYARI DATALOADER
# ============================================================

dataloader = (
    create_pretraining_dataloader(
        batch_size=BATCH_SIZE,
    )
)


# ============================================================
# 10. START TRAINING
# ============================================================

step = start_step

last_loss = None


for batch_number, (
    input_ids,
    labels,
) in enumerate(
    dataloader,
    start=1,
):


    # ========================================================
    # Skip batches already used before the checkpoint
    # ========================================================

    if batch_number <= start_step:

        continue


    # Current training step
    step = batch_number


    # ========================================================
    # Move batch to CPU / GPU
    # ========================================================

    input_ids = input_ids.to(
        device
    )

    labels = labels.to(
        device
    )


    # ========================================================
    # Remove gradients from previous step
    # ========================================================

    optimizer.zero_grad(
        set_to_none=True
    )


    # ========================================================
    # Forward pass
    # ========================================================

    logits = model(
        input_ids
    )


    # input_ids shape:
    #
    # [B, 1024]
    #
    #
    # logits shape:
    #
    # [B, 1024, 16000]


    # ========================================================
    # Cross Entropy Loss
    # ========================================================

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            VOCAB_SIZE,
        ),
        labels.reshape(-1),
    )


    last_loss = loss.item()


    # ========================================================
    # Backward pass
    # ========================================================

    loss.backward()


    # ========================================================
    # Gradient clipping
    # ========================================================

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
    )


    # ========================================================
    # Update VoyariLM weights
    # ========================================================

    optimizer.step()


    # ========================================================
    # Print training progress
    # ========================================================

    if step % LOG_EVERY == 0:

        print(
            f"Step {step:5d} | "
            f"Loss {loss.item():.4f} | "
            f"Grad Norm "
            f"{gradient_norm.item():.4f}"
        )


    # ========================================================
    # Periodic checkpoint saving
    # ========================================================

    if step % SAVE_EVERY == 0:

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"voyari_step_{step}.pt"
        )

        torch.save(
            {
                "step":
                    step,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss":
                    loss.item(),
            },
            checkpoint_path,
        )

        print()
        print(
            "Checkpoint saved:",
            checkpoint_path,
        )
        print()


    # ========================================================
    # Stop at MAX_STEPS
    # ========================================================

    if step >= MAX_STEPS:

        break


# ============================================================
# 11. SAVE FINAL CHECKPOINT
# ============================================================

if last_loss is not None:

    final_checkpoint_path = (
        CHECKPOINT_DIR
        / f"voyari_step_{step}.pt"
    )

    torch.save(
        {
            "step":
                step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                last_loss,
        },
        final_checkpoint_path,
    )

    print()
    print(
        "Final checkpoint saved:",
        final_checkpoint_path,
    )


# ============================================================
# 12. TRAINING FINISHED
# ============================================================

print()
print("=" * 60)

print(
    "VOYARILM PRETRAINING TEST COMPLETED"
)

print("=" * 60)

print(
    "Steps completed:",
    step,
)

if last_loss is not None:

    print(
        "Final loss:",
        last_loss,
    )