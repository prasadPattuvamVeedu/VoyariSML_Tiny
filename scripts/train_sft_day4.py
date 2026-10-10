"""VoyariLM Tiny Day 4 controlled SFT candidate from frozen SFT v2 step 100.

Reuses the proven assistant-only loss and optimizer loop from train_sft_v2.py.
The Day 3 parent checkpoint and training JSONLs are read-only. Day 4 uses
a distinct sft_day4 directory, stage and checkpoint prefix.

Recommended first run: --max-steps 100 --session-steps 25 --peak-lr 3e-6
--min-lr 8e-7 --warmup-steps 8 --grad-accum 8.
Always run --dry-run first and supply --holdout-file for all eval questions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.model_config import CONTEXT_LENGTH, VOCAB_SIZE
from src.data.sft_dataset import (
    FORMAT_VERSION,
    SFTCorpus,
    SFTSplitDataset,
    sha256_file,
)
from src.model.voyari_lm import VoyariLM


def parse_args():
    parser = argparse.ArgumentParser(description="VoyariLM Tiny Day 4 controlled SFT")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--training-file", action="append", type=Path, required=True,
                        help="Repeat for Day 4 generated and reviewed Day 3 replay JSONLs")
    parser.add_argument("--holdout-file", action="append", type=Path, default=[],
                        help="Exclude matching user questions from any training JSONL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate sources, checkpoint, configuration; do not train")
    parser.add_argument(
        "--tokenizer", type=Path,
        default=ROOT / "artifacts/tokenizer/voyari_tokenizer_16k_v3.json",
    )
    parser.add_argument("--run-name", default="grounded_mix_v1")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--session-steps", type=int, default=25,
                        help="Number of optimizer updates to attempt in this Kaggle session")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=3e-6)
    parser.add_argument("--min-lr", type=float, default=8e-7)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--keep-last", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto",
                        help="auto, none, or exact .pt SFT checkpoint file")
    return parser.parse_args()


def validate(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("run-name must contain only letters, digits, '_' and '-'")
    for name in ("max_steps", "session_steps", "grad_accum", "eval_samples",
                 "save_every", "eval_every", "log_every", "keep_last"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < args.min_lr <= args.peak_lr:
        raise ValueError("Learning rates must satisfy 0 < min_lr <= peak_lr")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("warmup_steps must be >= 0 and < max_steps")
    if args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("Invalid weight decay / max gradient norm")
    if args.max_length > CONTEXT_LENGTH:
        raise ValueError(f"max-length exceeds model context {CONTEXT_LENGTH}")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    files = [p.resolve() for p in args.training_file]
    if len(files) < 2 or len(files) != len(set(files)):
        raise ValueError("Use two or more DISTINCT training files (Day 4 + Day 3 replay)")
    for path in args.training_file + args.holdout_file:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.resolve() == args.base_checkpoint.resolve():
            raise ValueError("Training files cannot be the parent checkpoint")
    if any(path.resolve() in set(files) for path in args.holdout_file):
        raise ValueError("Training file must not also be a holdout")
    if any("diagnostic" in p.name.lower() or "eval" in p.name.lower()
           for p in args.training_file):
        raise ValueError("Evaluation and diagnostic files must NEVER be used for training")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)


def ordered_for_epoch(size, epoch, seed):
    # Stateless permutation: resume can reconstruct it from consumed micro-batches.
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def lr_at_step(step, total_steps, warmup_steps, peak_lr, min_lr):
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    if total_steps <= warmup_steps + 1:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def latest_checkpoint(directory):
    paths = []
    for path in directory.glob("voyari_day4_step_*.pt"):
        match = re.fullmatch(r"voyari_day4_step_(\d+)\.pt", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return max(paths)[1] if paths else None


def resolve_resume(value, directory):
    if value.lower() == "auto":
        return latest_checkpoint(directory)
    if value.lower() == "none":
        if latest_checkpoint(directory):
            raise RuntimeError(
                "SFT checkpoints already exist for this run name. "
                "Use --resume auto or choose a NEW --run-name."
            )
        return None
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.parent != directory.resolve():
        raise ValueError("Explicit resume checkpoint must belong to this run directory")
    return path


def loss_for_sample(model, sample, device, autocast_context):
    x, y = sample
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    with autocast_context():
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            ignore_index=-100,
        )
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite assistant loss")
    return loss


@torch.no_grad()
def evaluate(model, dataset, device, autocast_context, seed, sample_count):
    was_training = model.training
    model.eval()
    picks = random.Random(seed + 10_000).sample(
        range(len(dataset)), min(sample_count, len(dataset))
    )
    total_loss, total_tokens = 0.0, 0
    for idx in picks:
        x, y = dataset[idx]
        targets = int((y != -100).sum().item())
        loss = loss_for_sample(model, (x, y), device, autocast_context)
        total_loss += float(loss.item()) * targets
        total_tokens += targets
    model.train(was_training)
    return total_loss / max(total_tokens, 1)


def save_checkpoint(path, model, optimizer, scaler, global_step,
                    micro_seen, run_config, last_loss):
    payload = {
        "stage": "sft_day4",
        "parent_stage": "sft_v2",
        "parent_sft_step": 100,
        "step": global_step,
        "base_pretraining_step": 12014,
        "micro_batches_seen": micro_seen,
        "loss": last_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "torch_rng_state": torch.get_rng_state(),
        "run_config": run_config,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # Atomic replacement avoids partial .pt checkpoints.


def prune_old(directory, keep_last):
    checkpoints = sorted(
        directory.glob("voyari_day4_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    for old in checkpoints[:-keep_last]:
        old.unlink()


def main():
    args = parse_args()
    validate(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not use_fp16 else torch.float16
    def autocast_context():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError("Tokenizer and model vocabulary sizes differ")
    paths = args.training_file
    corpus = SFTCorpus(paths, tokenizer, args.max_length, args.val_fraction, args.seed)
    train_ds = SFTSplitDataset(corpus, "train")
    val_ds = SFTSplitDataset(corpus, "val")

    run_config = {
        "format_version": FORMAT_VERSION,
        "parent_stage": "sft_v2",
        "parent_sft_step": 100,
        "parent_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "data_sha256": corpus.file_hashes,
        "training_file_names": [p.name for p in paths],
        "holdout_file_sha256": {p.name: sha256_file(p) for p in args.holdout_file},
        "base_checkpoint_name": args.base_checkpoint.name,
        "base_checkpoint_bytes": args.base_checkpoint.stat().st_size,
        "max_length": args.max_length,
        "grad_accum": args.grad_accum,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "peak_lr": args.peak_lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
    }

    # Guard against accidentally training on test/evaluation prompts.
    if not args.holdout_file:
        raise ValueError("Supply --holdout-file for both evaluation datasets")
    if len({p.name for p in args.holdout_file}) != len(args.holdout_file):
        raise ValueError("Holdout filenames must be unique for config hashing")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Training basenames must be unique for config hashing")

    def user_prompts(file_path):
        with file_path.open(encoding="utf-8") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if isinstance(record.get("question"), str):
                    yield " ".join(record["question"].casefold().split())
                for message in record.get("messages", []):
                    if message.get("role") == "user":
                        yield " ".join(str(message.get("content", "")).casefold().split())

    forbidden = {q for path in args.holdout_file for q in user_prompts(path)}
    seen_train = {q for path in paths for q in user_prompts(path)}
    clashes = forbidden.intersection(seen_train)
    if clashes:
        raise RuntimeError(
            "Training/holdout prompt overlap: " + repr(sorted(clashes)[:3])
        )

    # Frozen SFT v2 model is the ONLY approved initial checkpoint.
    def validate_parent_checkpoint(checkpoint):
        if (checkpoint.get("stage") != "sft_v2"
                or int(checkpoint.get("step", -1)) != 100
                or checkpoint.get("run_config", {}).get("parent_sft_step") != 8000):
            raise RuntimeError("Day 4 parent must be SFT v2 step 100 from SFT v1 step 8000")

    if args.dry_run:
        checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
        validate_parent_checkpoint(checkpoint)
        print("DAY 4 TRAINING PREFLIGHT PASSED")
        print("Parent:", args.base_checkpoint, "| SFT v2 step 100")
        print("Training files:", [str(p) for p in paths])
        print("Train/val conversation counts:", corpus.conversations)
        print("Assistant examples:", {part: len(corpus.indices[part]) for part in ("train", "val")})
        print("Forbidden holdout overlap:", len(clashes))
        print("New run name:", args.run_name)
        print("Run config:", json.dumps(run_config, indent=2))
        corpus.close()
        return

    save_dir = ROOT / "artifacts/checkpoints/sft_day4" / args.run_name
    logs = ROOT / "logs/sft_day4"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    resume_file = resolve_resume(args.resume, save_dir)

    print("=" * 70)
    print("VOYARILM TINY - DAY 4 CONTROLLED SFT")
    print("=" * 70)
    print("Device:", device, "| Precision:",
          "FP16" if use_fp16 else ("BF16" if device.type == "cuda" else "FP32"))
    print("Tokenizer:", args.tokenizer.name)
    print("Run name:", args.run_name)
    print("Conversations:", corpus.conversations)
    print("Assistant examples (long outputs chunked):",
          {part: len(corpus.indices[part]) for part in ("train", "val")})
    print("Read-only SFT v2 step-100 parent:", args.base_checkpoint)

    model = VoyariLM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16, init_scale=1024.0)
    step = 0
    micro_seen = 0

    if resume_file:
        print("Resuming Day 4 checkpoint:", resume_file)
        checkpoint = torch.load(resume_file, map_location="cpu", weights_only=True)
        if checkpoint.get("stage") != "sft_day4" or checkpoint.get("run_config") != run_config:
            raise ValueError("Day 4 resume stage/config/data mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        step = int(checkpoint["step"])
        micro_seen = int(checkpoint["micro_batches_seen"])
        del checkpoint
    else:
        print("Initializing Day 4 from SFT v2 step 100 (fresh optimizer)")
        checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
        validate_parent_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint

    if step >= args.max_steps:
        print("Target step already reached:", step)
        return

    session_stop = min(args.max_steps, step + args.session_steps)
    print("Starting SFT step:", step, "| This session ends at:", session_stop)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Saving checkpoints to:", save_dir)
    print(flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_path = logs / f"{args.run_name}.jsonl"
    order_epoch = -1
    order = []
    start_time = time.monotonic()
    last_loss = None
    last_save_step = None
    while step < session_stop:
        # Set LR for the NEXT optimizer update, including after a resume.
        current_lr = lr_at_step(
            step + 1, args.max_steps, args.warmup_steps,
            args.peak_lr, args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            epoch, pos = divmod(micro_seen, len(train_ds))
            if epoch != order_epoch:
                order = ordered_for_epoch(len(train_ds), epoch, args.seed)
                order_epoch = epoch
            sample = train_ds[order[pos]]
            loss = loss_for_sample(model, sample, device, autocast_context)
            accumulated_loss += float(loss.detach().item())
            if scaler.is_enabled():
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()
            micro_seen += 1

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        old_scale = scaler.get_scale() if scaler.is_enabled() else None
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            if not torch.isfinite(grad_norm).item():
                raise RuntimeError("Non-finite gradients")
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled() and scaler.get_scale() < old_scale:
            print("FP16 overflow: optimizer step skipped; scale reduced", flush=True)
            continue

        step += 1
        last_loss = accumulated_loss / args.grad_accum
        if step == 1 or step % args.log_every == 0 or step == session_stop:
            print(f"SFT Step {step}/{args.max_steps} | Loss {last_loss:.4f} | "
                  f"Grad {float(grad_norm):.3f} | LR {current_lr:.2e} | "
                  f"Elapsed {time.monotonic()-start_time:.1f}s", flush=True)

        metric = {
            "step": step, "train_loss": last_loss,
            "grad_norm": float(grad_norm), "lr": current_lr,
            "micro_batches_seen": micro_seen,
        }
        if step % args.eval_every == 0 or step == session_stop:
            metric["val_loss"] = evaluate(
                model, val_ds, device, autocast_context, args.seed, args.eval_samples
            )
            print(f"Validation loss: {metric['val_loss']:.4f}", flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric) + "\n")

        if step % args.save_every == 0 or step == session_stop:
            path = save_dir / f"voyari_day4_step_{step:06d}.pt"
            save_checkpoint(path, model, optimizer, scaler, step, micro_seen,
                            run_config, last_loss)
            last_save_step = step
            prune_old(save_dir, args.keep_last)
            print("Saved SFT checkpoint:", path, flush=True)

    print("SFT Day 4 session complete | Final step:", step)
    print("Resume a new Kaggle session using SAME config and --resume auto")
    print("Metrics:", log_path)
    if last_save_step != step:
        raise RuntimeError("Final step not checkpointed")
    corpus.close()


if __name__ == "__main__":
    main()
