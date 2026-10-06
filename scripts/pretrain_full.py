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


FULL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "full_pretraining"
)

STAGE1_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "stage1"
    / "voyari_stage1_step_000200.pt"
)

LOG_DIR = PROJECT_ROOT / "logs"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "VoyariLM Tiny full pretraining - first corpus pass"
        )
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=12_014,
        help=(
            "Global optimizer step at the end of the first "
            "full corpus pass."
        ),
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
        "--warmup-steps",
        type=int,
        default=200,
        help=(
            "Full-pretraining re-warmup updates from the "
            "Stage-1 ending LR to peak LR."
        ),
    )

    parser.add_argument(
        "--peak-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--keep-last",
        type=int,
        default=2,
        help=(
            "Number of full-pretraining checkpoints to retain."
        ),
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
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help=(
            "auto = latest full checkpoint, otherwise Stage-1 "
            "step 200; stage1 = force Stage-1 step 200; "
            "or provide a checkpoint path."
        ),
    )

    parser.add_argument(
        "--session-steps",
        type=int,
        default=0,
        help=(
            "Optional number of optimizer updates to run in this "
            "session. 0 means continue to target-step."
        ),
    )

    return parser.parse_args()


def validate_args(args):

    positive_fields = {
        "target_step": args.target_step,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "save_every": args.save_every,
        "keep_last": args.keep_last,
        "log_every": args.log_every,
    }

    for name, value in positive_fields.items():

        if value <= 0:

            raise ValueError(
                f"{name} must be greater than 0."
            )

    if args.warmup_steps < 0:

        raise ValueError(
            "warmup_steps cannot be negative."
        )

    if args.peak_lr <= 0:

        raise ValueError(
            "peak_lr must be greater than 0."
        )

    if args.min_lr <= 0:

        raise ValueError(
            "min_lr must be greater than 0."
        )

    if args.min_lr > args.peak_lr:

        raise ValueError(
            "min_lr cannot exceed peak_lr."
        )

    if args.max_grad_norm <= 0:

        raise ValueError(
            "max_grad_norm must be greater than 0."
        )

    if args.session_steps < 0:

        raise ValueError(
            "session_steps cannot be negative."
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


def find_latest_full_checkpoint():

    if not FULL_CHECKPOINT_DIR.exists():

        return None

    pattern = re.compile(
        r"voyari_full_step_(\d+)\.pt$"
    )

    candidates = []

    for path in FULL_CHECKPOINT_DIR.glob(
        "voyari_full_step_*.pt"
    ):

        match = pattern.search(path.name)

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

    if normalized == "auto":

        latest = find_latest_full_checkpoint()

        if latest is not None:
            return latest

        if not STAGE1_CHECKPOINT.is_file():

            raise FileNotFoundError(
                "No full-pretraining checkpoint was found, "
                "and the Stage-1 step-200 checkpoint is missing:\n"
                f"{STAGE1_CHECKPOINT}"
            )

        return STAGE1_CHECKPOINT

    if normalized == "stage1":

        if not STAGE1_CHECKPOINT.is_file():

            raise FileNotFoundError(
                "Stage-1 step-200 checkpoint was not found:\n"
                f"{STAGE1_CHECKPOINT}"
            )

        return STAGE1_CHECKPOINT

    path = Path(value)

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    if not path.is_file():

        raise FileNotFoundError(
            f"Resume checkpoint not found: {path}"
        )

    return path


def validate_resume_batching(
    checkpoint,
    batch_size,
    grad_accum,
):

    saved_args = checkpoint.get(
        "args",
        {}
    )

    saved_batch_size = saved_args.get(
        "batch_size"
    )

    saved_grad_accum = saved_args.get(
        "grad_accum"
    )

    if (
        saved_batch_size is not None
        and int(saved_batch_size) != batch_size
    ):

        raise ValueError(
            "batch_size must match the checkpoint. "
            f"Checkpoint={saved_batch_size}, "
            f"requested={batch_size}."
        )

    if (
        saved_grad_accum is not None
        and int(saved_grad_accum) != grad_accum
    ):

        raise ValueError(
            "grad_accum must match the checkpoint. "
            f"Checkpoint={saved_grad_accum}, "
            f"requested={grad_accum}."
        )


def build_schedule(
    checkpoint,
    args,
    global_step,
    optimizer,
):

    checkpoint_stage = checkpoint.get(
        "stage",
        ""
    )

    if checkpoint_stage == "full_pretraining":

        schedule = checkpoint.get(
            "schedule"
        )

        if not schedule:

            raise ValueError(
                "Full-pretraining checkpoint is missing "
                "its saved schedule."
            )

        checks = {
            "target_step": args.target_step,
            "warmup_steps": args.warmup_steps,
            "peak_lr": args.peak_lr,
            "min_lr": args.min_lr,
        }

        for name, requested in checks.items():

            saved = schedule[name]

            if isinstance(requested, float):

                matches = math.isclose(
                    float(saved),
                    float(requested),
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )

            else:

                matches = int(saved) == int(requested)

            if not matches:

                raise ValueError(
                    "Full-pretraining schedule does not match "
                    f"the checkpoint for {name}: "
                    f"checkpoint={saved}, requested={requested}."
                )

        return schedule

    if checkpoint_stage != "stage1_stability":

        raise ValueError(
            "Expected a Stage-1 or full-pretraining "
            f"checkpoint, found stage={checkpoint_stage!r}."
        )

    start_lr = float(
        optimizer.param_groups[0]["lr"]
    )

    schedule = {
        "phase_start_step": global_step,
        "target_step": args.target_step,
        "start_lr": start_lr,
        "peak_lr": args.peak_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
    }

    total_phase_updates = (
        args.target_step
        - global_step
    )

    if total_phase_updates <= 0:

        raise ValueError(
            "target_step must be greater than the "
            "Stage-1 checkpoint step."
        )

    if args.warmup_steps >= total_phase_updates:

        raise ValueError(
            "warmup_steps must be smaller than the "
            "number of full-pretraining updates."
        )

    return schedule


def learning_rate_for_step(
    step,
    schedule,
):

    phase_start_step = int(
        schedule["phase_start_step"]
    )

    target_step = int(
        schedule["target_step"]
    )

    warmup_steps = int(
        schedule["warmup_steps"]
    )

    start_lr = float(
        schedule["start_lr"]
    )

    peak_lr = float(
        schedule["peak_lr"]
    )

    min_lr = float(
        schedule["min_lr"]
    )

    phase_step = (
        step
        - phase_start_step
    )

    total_updates = (
        target_step
        - phase_start_step
    )

    if phase_step <= 0:

        return start_lr

    if warmup_steps > 0 and phase_step <= warmup_steps:

        if warmup_steps == 1:
            return peak_lr

        warmup_progress = (
            (phase_step - 1)
            / (warmup_steps - 1)
        )

        return (
            start_lr
            + (peak_lr - start_lr)
            * warmup_progress
        )

    decay_updates = (
        total_updates
        - warmup_steps
    )

    if decay_updates <= 0:

        return min_lr

    if warmup_steps == 0:

        if total_updates == 1:
            progress = 1.0
        else:
            progress = (
                (phase_step - 1)
                / (total_updates - 1)
            )

    else:

        progress = (
            (phase_step - warmup_steps)
            / decay_updates
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
                math.pi * progress
            )
        )
    )

    return (
        min_lr
        + (peak_lr - min_lr)
        * cosine
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
    schedule,
):

    payload = {
        "stage": "full_pretraining",
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
        "schedule": schedule,
    }

    if torch.cuda.is_available():

        payload["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        path
    )


def prune_old_checkpoints(keep_last):

    pattern = re.compile(
        r"voyari_full_step_(\d+)\.pt$"
    )

    checkpoints = []

    for path in FULL_CHECKPOINT_DIR.glob(
        "voyari_full_step_*.pt"
    ):

        match = pattern.search(path.name)

        if match:

            checkpoints.append(
                (
                    int(match.group(1)),
                    path,
                )
            )

    checkpoints.sort(
        key=lambda item: item[0]
    )

    for _, path in checkpoints[:-keep_last]:

        path.unlink()

        print(
            "Removed old checkpoint:",
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


def restore_rng_state(
    checkpoint,
    device,
):

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


def checkpoint_path_for_step(step):

    return (
        FULL_CHECKPOINT_DIR
        / (
            "voyari_full_step_"
            f"{step:06d}.pt"
        )
    )


def main():

    args = parse_args()
    validate_args(args)

    FULL_CHECKPOINT_DIR.mkdir(
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    resume_path = resolve_resume_checkpoint(
        args.resume
    )

    print()
    print(
        "Loading checkpoint:",
        resume_path,
    )

    checkpoint = torch.load(
        resume_path,
        map_location=device,
    )

    validate_resume_batching(
        checkpoint,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
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
            global_step * args.grad_accum,
        )
    )

    tokens_seen = int(
        checkpoint.get(
            "tokens_seen",
            (
                global_step
                * args.grad_accum
                * args.batch_size
                * CONTEXT_LENGTH
            ),
        )
    )

    last_loss = checkpoint.get(
        "loss"
    )

    schedule = build_schedule(
        checkpoint=checkpoint,
        args=args,
        global_step=global_step,
        optimizer=optimizer,
    )

    restore_rng_state(
        checkpoint,
        device,
    )

    if global_step >= args.target_step:

        print()
        print(
            "Checkpoint already reached target step "
            f"{args.target_step}."
        )

        return

    checkpoint_stage = checkpoint.get(
        "stage",
        ""
    )

    metrics_path = (
        LOG_DIR
        / "full_pretraining_metrics.jsonl"
    )

    if (
        checkpoint_stage == "stage1_stability"
        and metrics_path.exists()
    ):

        metrics_path.unlink()

    session_target_step = args.target_step

    if args.session_steps > 0:

        session_target_step = min(
            args.target_step,
            global_step + args.session_steps,
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.train()

    dataloader = create_pretraining_dataloader(
        batch_size=args.batch_size,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats()

    print()
    print("=" * 76)
    print("VOYARILM TINY - FULL PRETRAINING / FIRST CORPUS PASS")
    print("=" * 76)
    print("Device                 :", device)
    print("AMP                    :", (
        "bf16"
        if use_bf16
        else (
            "fp16"
            if use_fp16
            else "off"
        )
    ))
    print("Trainable parameters   :", f"{trainable_parameters:,}")
    print("Vocabulary size        :", f"{VOCAB_SIZE:,}")
    print("Context length         :", f"{CONTEXT_LENGTH:,}")
    print("BOS / EOS IDs          :", BOS_ID, "/", EOS_ID)
    print("Batch size             :", args.batch_size)
    print("Gradient accumulation  :", args.grad_accum)
    print(
        "Tokens / update        :",
        (
            args.batch_size
            * args.grad_accum
            * CONTEXT_LENGTH
        ),
    )
    print("Loaded checkpoint stage:", checkpoint_stage)
    print("Starting global step   :", global_step)
    print("Target global step     :", args.target_step)
    print("Session target step    :", session_target_step)
    print("Micro-batches to skip  :", f"{micro_batches_seen:,}")
    print("Schedule start LR      :", f"{schedule['start_lr']:.2e}")
    print("Schedule peak LR       :", f"{schedule['peak_lr']:.2e}")
    print("Schedule min LR        :", f"{schedule['min_lr']:.2e}")
    print("Schedule warmup updates:", schedule["warmup_steps"])
    print("=" * 76)

    accumulation_count = 0
    accumulated_loss = 0.0
    session_tokens = 0
    started_at = None
    announced_resume = False

    for batch_number, (
        input_ids,
        labels,
    ) in enumerate(
        dataloader,
        start=1,
    ):

        if batch_number <= micro_batches_seen:
            continue

        if not announced_resume:

            if micro_batches_seen > 0:

                print()
                print(
                    "Resume position reached. "
                    f"Continuing from micro-batch "
                    f"{micro_batches_seen + 1:,}."
                )

            announced_resume = True
            started_at = time.time()

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

        next_step = global_step + 1

        current_lr = learning_rate_for_step(
            step=next_step,
            schedule=schedule,
        )

        for parameter_group in optimizer.param_groups:

            parameter_group["lr"] = (
                current_lr
            )

        if scaler.is_enabled():

            scaler.unscale_(
                optimizer
            )

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
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
            "session_tokens": session_tokens,
            "tokens_per_second": tokens_per_second,
            "max_cuda_memory_gb": max_memory_gb,
        }

        append_metric(
            metrics_path,
            metric,
        )

        if (
            global_step == (
                int(schedule["phase_start_step"])
                + 1
            )
            or global_step % args.log_every == 0
        ):

            print(
                f"Step {global_step:6d} | "
                f"Loss {step_loss:8.4f} | "
                f"Grad {gradient_norm.item():7.3f} | "
                f"LR {current_lr:.2e} | "
                f"Tok/s {tokens_per_second:8.0f} | "
                f"VRAM {max_memory_gb:5.2f} GB"
            )

        if global_step % args.save_every == 0:

            checkpoint_path = (
                checkpoint_path_for_step(
                    global_step
                )
            )

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                micro_batches_seen=micro_batches_seen,
                tokens_seen=tokens_seen,
                loss=step_loss,
                args=args,
                schedule=schedule,
            )

            prune_old_checkpoints(
                args.keep_last
            )

            print(
                "Saved checkpoint:",
                checkpoint_path,
            )

        if global_step >= session_target_step:
            break

    if started_at is None:

        raise RuntimeError(
            "No new training batch was available after "
            "the saved resume position."
        )

    if global_step < session_target_step:

        raise RuntimeError(
            "The pretraining stream ended before the requested "
            f"session target. Last completed step={global_step}, "
            f"session target={session_target_step}."
        )

    final_checkpoint = (
        checkpoint_path_for_step(
            global_step
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
        schedule=schedule,
    )

    prune_old_checkpoints(
        args.keep_last
    )

    print()
    print("=" * 76)

    if global_step >= args.target_step:

        print("FULL PRETRAINING FIRST PASS COMPLETED")

    else:

        print("FULL PRETRAINING SESSION COMPLETED")
        print(
            "Run the same command again with "
            "--resume auto to continue."
        )

    print("=" * 76)
    print("Final global step :", global_step)
    print("Final loss        :", f"{last_loss:.4f}")
    print("Tokens seen       :", f"{tokens_seen:,}")
    print("Checkpoint        :", final_checkpoint)
    print("Metrics           :", metrics_path)


if __name__ == "__main__":

    main()
