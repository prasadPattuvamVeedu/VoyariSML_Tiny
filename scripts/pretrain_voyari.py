import math

import torch
import torch.nn.functional as F

from configs.model_config import VOCAB_SIZE

from src.data.voyari_data_config import (
    PROJECT_ROOT,
)

from src.data.pretraining_dataset import (
    create_pretraining_dataloader,
)

from src.model.voyari_lm import VoyariLM


# ============================================================
# 1. TRAINING SETTINGS
# ============================================================

# Number of 1024-token sequences loaded at once.
BATCH_SIZE = 1

# Accumulate gradients across several batches before
# updating the model weights.
GRAD_ACCUM_STEPS = 8

# Maximum learning rate.
MAX_LEARNING_RATE = 3e-4

# Learning rate at the end of cosine decay.
MIN_LEARNING_RATE = 3e-5

# Gradually increase learning rate during the first steps.
WARMUP_STEPS = 20

# Stage-1 stability test.
# This means 200 optimizer updates.
MAX_STEPS = 200

# Print training information every optimizer step.
LOG_EVERY = 1

# Save a checkpoint every 50 optimizer steps.
SAVE_EVERY = 50

# Maximum allowed gradient norm.
MAX_GRAD_NORM = 1.0

# Reproducibility.
SEED = 42


# ============================================================
# 2. CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. OPTIONAL RESUME CHECKPOINT
# ============================================================

# Stage 1 should start from a fresh model.
#
# Later, if you want to resume, change this to something like:
#
# RESUME_CHECKPOINT = (
#     CHECKPOINT_DIR
#     / "voyari_stage1_step_100.pt"
# )
#
RESUME_CHECKPOINT = None


# ============================================================
# 4. RANDOM SEED
# ============================================================

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("VOYARILM STAGE-1 PRETRAINING")
print("=" * 70)

print()
print("Using device:", device)


# ============================================================
# 6. MIXED PRECISION MODE
# ============================================================

USE_AMP = (
    device.type == "cuda"
)

USE_BF16 = (
    USE_AMP
    and torch.cuda.is_bf16_supported()
)

USE_FP16 = (
    USE_AMP
    and not USE_BF16
)


if USE_BF16:

    AMP_DTYPE = torch.bfloat16

elif USE_FP16:

    AMP_DTYPE = torch.float16

else:

    AMP_DTYPE = None


print(
    "Mixed precision:",
    (
        "BF16"
        if USE_BF16
        else "FP16"
        if USE_FP16
        else "Disabled"
    ),
)


# FP16 needs gradient scaling.
# BF16 normally does not.
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_FP16,
)


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = VoyariLM().to(
    device
)


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print()
print(
    "Total parameters:",
    f"{total_parameters:,}",
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}",
)


# ============================================================
# 8. CREATE OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=MAX_LEARNING_RATE,
)


# ============================================================
# 9. LEARNING-RATE SCHEDULE
# ============================================================

def get_learning_rate(step):
    """
    Warmup followed by cosine decay.

    step starts at 1.
    """

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    if step <= WARMUP_STEPS:

        warmup_ratio = (
            step
            / WARMUP_STEPS
        )

        return (
            MAX_LEARNING_RATE
            * warmup_ratio
        )


    # --------------------------------------------------------
    # Cosine decay
    # --------------------------------------------------------

    decay_steps = (
        MAX_STEPS
        - WARMUP_STEPS
    )

    progress = (
        step
        - WARMUP_STEPS
    ) / decay_steps

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine_value = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    learning_rate = (
        MIN_LEARNING_RATE
        + (
            MAX_LEARNING_RATE
            - MIN_LEARNING_RATE
        )
        * cosine_value
    )

    return learning_rate


# ============================================================
# 10. CHECKPOINT SAVING FUNCTION
# ============================================================

def save_checkpoint(
    optimizer_step,
    micro_batches_seen,
    loss_value,
):

    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            f"voyari_stage1_step_"
            f"{optimizer_step}.pt"
        )
    )

    checkpoint = {

        "optimizer_step":
            optimizer_step,

        "micro_batches_seen":
            micro_batches_seen,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "loss":
            loss_value,

        "training_config": {

            "batch_size":
                BATCH_SIZE,

            "gradient_accumulation_steps":
                GRAD_ACCUM_STEPS,

            "max_learning_rate":
                MAX_LEARNING_RATE,

            "min_learning_rate":
                MIN_LEARNING_RATE,

            "warmup_steps":
                WARMUP_STEPS,

            "max_steps":
                MAX_STEPS,
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print()
    print(
        "Checkpoint saved:",
        checkpoint_path,
    )
    print()

    return checkpoint_path


# ============================================================
# 11. RESUME STATE
# ============================================================

optimizer_step = 0

resume_micro_batches = 0

last_loss = None


if RESUME_CHECKPOINT is not None:

    if not RESUME_CHECKPOINT.is_file():

        raise FileNotFoundError(
            "Resume checkpoint was not found:\n"
            f"{RESUME_CHECKPOINT}"
        )


    print()
    print(
        "Loading checkpoint:",
        RESUME_CHECKPOINT,
    )


    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=device,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    if (
        USE_FP16
        and "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )


    optimizer_step = checkpoint[
        "optimizer_step"
    ]

    resume_micro_batches = checkpoint[
        "micro_batches_seen"
    ]

    last_loss = checkpoint.get(
        "loss"
    )


    print(
        "Resuming optimizer step:",
        optimizer_step,
    )

    print(
        "Micro-batches already used:",
        resume_micro_batches,
    )

    print(
        "Previous loss:",
        last_loss,
    )

else:

    print()
    print(
        "Starting Stage-1 from fresh model weights."
    )


# ============================================================
# 12. TRAINING MODE
# ============================================================

model.train()


# ============================================================
# 13. CREATE PRETRAINING DATALOADER
# ============================================================

dataloader = (
    create_pretraining_dataloader(
        batch_size=BATCH_SIZE,
    )
)


# ============================================================
# 14. EFFECTIVE BATCH INFORMATION
# ============================================================

effective_batch_size = (
    BATCH_SIZE
    * GRAD_ACCUM_STEPS
)

print()
print(
    "Physical batch size:",
    BATCH_SIZE,
)

print(
    "Gradient accumulation steps:",
    GRAD_ACCUM_STEPS,
)

print(
    "Effective batch size:",
    effective_batch_size,
)

print(
    "Approx. tokens per optimizer update:",
    effective_batch_size * 1024,
)

print()


# ============================================================
# 15. CLEAR INITIAL GRADIENTS
# ============================================================

optimizer.zero_grad(
    set_to_none=True
)


# ============================================================
# 16. TRAINING LOOP STATE
# ============================================================

accumulated_loss = 0.0

accumulation_count = 0

micro_batches_seen = (
    resume_micro_batches
)


# ============================================================
# 17. TRAINING LOOP
# ============================================================

for stream_batch_number, (
    input_ids,
    labels,
) in enumerate(
    dataloader,
    start=1,
):


    # --------------------------------------------------------
    # Resume:
    # skip micro-batches that were already trained on.
    # --------------------------------------------------------

    if (
        stream_batch_number
        <= resume_micro_batches
    ):

        continue


    micro_batches_seen = (
        stream_batch_number
    )


    # --------------------------------------------------------
    # Move batch to device
    # --------------------------------------------------------

    input_ids = input_ids.to(
        device,
        non_blocking=True,
    )

    labels = labels.to(
        device,
        non_blocking=True,
    )


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    if USE_AMP:

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
        ):

            logits = model(
                input_ids
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    VOCAB_SIZE,
                ),
                labels.reshape(
                    -1
                ),
            )

    else:

        logits = model(
            input_ids
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            labels.reshape(
                -1
            ),
        )


    # --------------------------------------------------------
    # Stop immediately if loss becomes NaN or Inf
    # --------------------------------------------------------

    if not torch.isfinite(loss):

        raise RuntimeError(
            "Training stopped because loss became "
            f"non-finite: {loss.item()}"
        )


    # Save original loss for reporting.
    batch_loss = loss.item()

    accumulated_loss += (
        batch_loss
    )

    accumulation_count += 1


    # --------------------------------------------------------
    # Divide loss because gradients are accumulated
    # over several micro-batches.
    # --------------------------------------------------------

    loss_for_backward = (
        loss
        / GRAD_ACCUM_STEPS
    )


    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    if USE_FP16:

        scaler.scale(
            loss_for_backward
        ).backward()

    else:

        loss_for_backward.backward()


    # --------------------------------------------------------
    # Keep accumulating until enough micro-batches
    # have been processed.
    # --------------------------------------------------------

    if (
        accumulation_count
        < GRAD_ACCUM_STEPS
    ):

        continue


    # ========================================================
    # ONE OPTIMIZER UPDATE STARTS HERE
    # ========================================================


    # --------------------------------------------------------
    # Set learning rate for the next optimizer step.
    # --------------------------------------------------------

    next_optimizer_step = (
        optimizer_step
        + 1
    )

    learning_rate = get_learning_rate(
        next_optimizer_step
    )

    for parameter_group in optimizer.param_groups:

        parameter_group["lr"] = (
            learning_rate
        )


    # --------------------------------------------------------
    # FP16 gradients must be unscaled before clipping.
    # --------------------------------------------------------

    if USE_FP16:

        scaler.unscale_(
            optimizer
        )


    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
        )
    )


    if not torch.isfinite(
        gradient_norm
    ):

        raise RuntimeError(
            "Training stopped because gradient "
            "norm became NaN or Inf."
        )


    # --------------------------------------------------------
    # Update model weights
    # --------------------------------------------------------

    if USE_FP16:

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        optimizer.step()


    optimizer_step += 1


    # --------------------------------------------------------
    # Average loss across accumulated micro-batches.
    # --------------------------------------------------------

    average_loss = (
        accumulated_loss
        / accumulation_count
    )

    last_loss = (
        average_loss
    )


    # --------------------------------------------------------
    # Clear gradients for the next accumulation cycle.
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulated_loss = 0.0

    accumulation_count = 0


    # ========================================================
    # LOGGING
    # ========================================================

    if (
        optimizer_step
        % LOG_EVERY
        == 0
    ):

        print(
            f"Step {optimizer_step:4d} | "
            f"Loss {average_loss:.4f} | "
            f"LR {learning_rate:.8f} | "
            f"Grad Norm "
            f"{gradient_norm.item():.4f}"
        )


    # ========================================================
    # PERIODIC CHECKPOINT
    # ========================================================

    if (
        optimizer_step
        % SAVE_EVERY
        == 0
    ):

        save_checkpoint(
            optimizer_step=
                optimizer_step,

            micro_batches_seen=
                micro_batches_seen,

            loss_value=
                average_loss,
        )


    # ========================================================
    # STOP STAGE-1
    # ========================================================

    if (
        optimizer_step
        >= MAX_STEPS
    ):

        break


# ============================================================
# 18. SAVE FINAL CHECKPOINT
# ============================================================

if last_loss is not None:

    final_checkpoint_path = (
        save_checkpoint(
            optimizer_step=
                optimizer_step,

            micro_batches_seen=
                micro_batches_seen,

            loss_value=
                last_loss,
        )
    )

else:

    final_checkpoint_path = None


# ============================================================
# 19. TRAINING FINISHED
# ============================================================

print()
print("=" * 70)
print("VOYARILM STAGE-1 PRETRAINING COMPLETED")
print("=" * 70)

print()
print(
    "Optimizer steps completed:",
    optimizer_step,
)

print(
    "Micro-batches processed:",
    micro_batches_seen,
)

if last_loss is not None:

    print(
        "Final loss:",
        f"{last_loss:.4f}",
    )

if final_checkpoint_path is not None:

    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )

print()