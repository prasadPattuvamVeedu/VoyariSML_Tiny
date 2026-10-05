from contextlib import nullcontext
from pathlib import Path
import argparse
import json
import math
import random
import re
import sys
import time

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from configs.model_config import (
    CONTEXT_LENGTH,
    VOCAB_SIZE,
)

from src.data.pretraining_dataset import (
    create_pretraining_dataloader,
)

from src.data.pretraining_documents import (
    PRETRAINING_SOURCES,
    check_file,
)

from src.data.pretraining_formatter import (
    BOS_ID,
    EOS_ID,
)

from src.model.voyari_lm import VoyariLM


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "stage1"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "VoyariLM Tiny Stage-1 pretraining stability run"
        )
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help=(
            "none = fresh run, "
            "auto = latest Stage-1 checkpoint, "
            "or provide a checkpoint path"
        ),
    )

    return parser.parse_args()


def validate_args(args):

    positive_fields = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "save_every": args.save_every,
        "log_every": args.log_every,
    }

    for name, value in positive_fields.items():

        if value <= 0:

            raise ValueError(
                f"{name} must be greater than 0."
            )

    if args.learning_rate <= 0:

        raise ValueError(
            "learning_rate must be greater than 0."
        )

    if args.warmup_steps < 0:

        raise ValueError(
            "warmup_steps cannot be negative."
        )

    if args.warmup_steps > args.max_steps:

        raise ValueError(
            "warmup_steps cannot exceed max_steps."
        )


def seed_everything(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def verify_training_files():

    print()
    print("Checking final pretraining corpora...")

    for source, path in PRETRAINING_SOURCES.items():

        check_file(path)

        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"  {source:28} "
            f"{size_mb:10.2f} MB"
        )


def find_latest_checkpoint():

    if not CHECKPOINT_DIR.exists():

        return None

    pattern = re.compile(
        r"voyari_stage1_step_(\d+)\.pt$"
    )

    candidates = []

    for path in CHECKPOINT_DIR.glob(
        "voyari_stage1_step_*.pt"
    ):

        match = pattern.search(
            path.name
        )

        if match:

            candidates.append(
                (
                    int(match.group(1)),
                    path,
                )
            )

    if not candidates:

        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[-1][1]


def resolve_resume_checkpoint(value):

    normalized = value.strip().lower()

    if normalized == "none":

        return None

    if normalized == "auto":

        return find_latest_checkpoint()

    path = Path(value)

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    if not path.exists():

        raise FileNotFoundError(
            f"Resume checkpoint not found: {path}"
        )

    return path


def learning_rate_for_step(
    step,
    max_steps,
    warmup_steps,
    base_lr,
    min_lr_ratio=0.10,
):

    if warmup_steps > 0 and step <= warmup_steps:

        return (
            base_lr
            * step
            / warmup_steps
        )

    if max_steps <= warmup_steps:

        return base_lr

    progress = (
        (step - warmup_steps)
        / (max_steps - warmup_steps)
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    multiplier = (
        min_lr_ratio
        + (
            1.0
            - min_lr_ratio
        )
        * cosine
    )

    return (
        base_lr
        * multiplier
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    step,
    micro_batches_seen,
    tokens_seen,
    loss,
    args,
):

    payload = {
        "stage": "stage1_stability",
        "step": step,
        "micro_batches_seen": micro_batches_seen,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler.is_enabled()
            else None
        ),
        "torch_rng_state": torch.get_rng_state(),
        "args": vars(args),
    }

    if torch.cuda.is_available():

        payload["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )

    torch.save(
        payload,
        path,
    )


def append_metric(path, metric):

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(metric)
            + "\n"
        )


def main():

    args = parse_args()
    validate_args(args)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    verify_training_files()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_bf16 = (
        device.type == "cuda"
        and hasattr(
            torch.cuda,
            "is_bf16_supported",
        )
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        device.type == "cuda"
        and not use_bf16
    )

    amp_dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16,
    )

    model = VoyariLM().to(
        device
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    resume_path = resolve_resume_checkpoint(
        args.resume
    )

    global_step = 0
    micro_batches_seen = 0
    tokens_seen = 0
    last_loss = None

    if resume_path is not None:

        print()
        print(
            "Loading Stage-1 checkpoint:",
            resume_path,
        )

        checkpoint = torch.load(
            resume_path,
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
            scaler.is_enabled()
            and checkpoint.get(
                "scaler_state_dict"
            )
        ):

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        global_step = int(
            checkpoint.get(
                "step",
                0,
            )
        )

        micro_batches_seen = int(
            checkpoint.get(
                "micro_batches_seen",
                (
                    global_step
                    * args.grad_accum
                ),
            )
        )

        tokens_seen = int(
            checkpoint.get(
                "tokens_seen",
                0,
            )
        )

        last_loss = checkpoint.get(
            "loss"
        )

        if "torch_rng_state" in checkpoint:

            torch.set_rng_state(
                checkpoint[
                    "torch_rng_state"
                ].cpu()
            )

        if (
            device.type == "cuda"
            and "cuda_rng_state_all"
            in checkpoint
        ):

            cuda_rng_states = [
                state.cpu()
                for state in checkpoint[
                    "cuda_rng_state_all"
                ]
            ]

            torch.cuda.set_rng_state_all(
                cuda_rng_states
            )

    if global_step >= args.max_steps:

        print()
        print(
            "Checkpoint already reached "
            f"step {global_step}, which is >= "
            f"requested max_steps={args.max_steps}."
        )

        return

    model.train()

    dataloader = create_pretraining_dataloader(
        batch_size=args.batch_size,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    metrics_path = (
        LOG_DIR
        / "stage1_metrics.jsonl"
    )

    print()
    print("=" * 72)
    print("VOYARILM TINY - STAGE 1 PRETRAINING")
    print("=" * 72)
    print("Device               :", device)
    print("AMP                  :", (
        "bf16"
        if use_bf16
        else (
            "fp16"
            if use_fp16
            else "off"
        )
    ))
    print("Trainable parameters :", f"{trainable_parameters:,}")
    print("Vocabulary size      :", f"{VOCAB_SIZE:,}")
    print("Context length       :", f"{CONTEXT_LENGTH:,}")
    print("BOS / EOS IDs        :", BOS_ID, "/", EOS_ID)
    print("Batch size           :", args.batch_size)
    print("Gradient accumulation:", args.grad_accum)
    print(
        "Tokens / update      :",
        (
            args.batch_size
            * args.grad_accum
            * CONTEXT_LENGTH
        ),
    )
    print("Starting step        :", global_step)
    print("Target step          :", args.max_steps)
    print("=" * 72)

    skipped = 0
    accumulation_count = 0
    accumulated_loss = 0.0
    session_tokens = 0
    started_at = time.time()

    for batch_number, (
        input_ids,
        labels,
    ) in enumerate(
        dataloader,
        start=1,
    ):

        if batch_number <= micro_batches_seen:

            skipped += 1
            continue

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if device.type == "cuda":

            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
            )

        else:

            autocast_context = nullcontext()

        with autocast_context:

            logits = model(
                input_ids
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    VOCAB_SIZE,
                ),
                labels.reshape(-1),
            )

            backward_loss = (
                loss
                / args.grad_accum
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss detected: {loss.item()}"
            )

        if scaler.is_enabled():

            scaler.scale(
                backward_loss
            ).backward()

        else:

            backward_loss.backward()

        accumulation_count += 1
        accumulated_loss += loss.item()

        micro_batches_seen = batch_number

        batch_tokens = input_ids.numel()
        tokens_seen += batch_tokens
        session_tokens += batch_tokens

        if accumulation_count < args.grad_accum:

            continue

        next_step = (
            global_step
            + 1
        )

        current_lr = learning_rate_for_step(
            step=next_step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.learning_rate,
        )

        for param_group in optimizer.param_groups:

            param_group["lr"] = (
                current_lr
            )

        if scaler.is_enabled():

            scaler.unscale_(
                optimizer
            )

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):

            raise RuntimeError(
                "Non-finite gradient norm detected."
            )

        if scaler.is_enabled():

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        global_step = next_step

        step_loss = (
            accumulated_loss
            / accumulation_count
        )

        last_loss = step_loss

        accumulation_count = 0
        accumulated_loss = 0.0

        elapsed = max(
            time.time() - started_at,
            1e-6,
        )

        tokens_per_second = (
            session_tokens
            / elapsed
        )

        max_memory_gb = 0.0

        if device.type == "cuda":

            max_memory_gb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

        metric = {
            "step": global_step,
            "loss": step_loss,
            "grad_norm": float(
                gradient_norm.item()
            ),
            "learning_rate": current_lr,
            "tokens_seen": tokens_seen,
            "tokens_per_second": (
                tokens_per_second
            ),
            "max_cuda_memory_gb": (
                max_memory_gb
            ),
        }

        append_metric(
            metrics_path,
            metric,
        )

        if (
            global_step == 1
            or global_step % args.log_every == 0
        ):

            print(
                f"Step {global_step:5d} | "
                f"Loss {step_loss:8.4f} | "
                f"Grad {gradient_norm.item():7.3f} | "
                f"LR {current_lr:.2e} | "
                f"Tok/s {tokens_per_second:8.0f} | "
                f"VRAM {max_memory_gb:5.2f} GB"
            )

        if (
            global_step % args.save_every == 0
        ):

            checkpoint_path = (
                CHECKPOINT_DIR
                / (
                    "voyari_stage1_step_"
                    f"{global_step:06d}.pt"
                )
            )

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                micro_batches_seen=(
                    micro_batches_seen
                ),
                tokens_seen=tokens_seen,
                loss=step_loss,
                args=args,
            )

            print(
                "Saved checkpoint:",
                checkpoint_path,
            )

        if global_step >= args.max_steps:

            break

    if global_step == 0:

        raise RuntimeError(
            "No optimizer step was completed. "
            "Check the dataset and resume position."
        )

    if global_step < args.max_steps:

        raise RuntimeError(
            "The pretraining stream ended before "
            f"max_steps={args.max_steps}. "
            f"Last completed step={global_step}."
        )

    final_checkpoint = (
        CHECKPOINT_DIR
        / (
            "voyari_stage1_step_"
            f"{global_step:06d}.pt"
        )
    )

    save_checkpoint(
        path=final_checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=global_step,
        micro_batches_seen=micro_batches_seen,
        tokens_seen=tokens_seen,
        loss=last_loss,
        args=args,
    )

    print()
    print("=" * 72)
    print("STAGE-1 STABILITY RUN COMPLETED")
    print("=" * 72)
    print("Final step :", global_step)
    print("Final loss :", f"{last_loss:.4f}")
    print("Tokens seen:", f"{tokens_seen:,}")
    print("Checkpoint :", final_checkpoint)
    print("Metrics    :", metrics_path)


if __name__ == "__main__":

    main()
