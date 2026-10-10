"""Back up Day 4 step-25 SFT candidate to a NEW private Kaggle Dataset.

Copies original checkpoint and associated artifacts; never edits their bytes,
never touches frozen Day 3 backup. Kaggle API CLI can mistakenly return exit=0
when dataset creation failed; therefore verify the remote file listing.
Usage: python scripts/backup_day4_checkpoint.py --checkpoint PATH
  --train PATH --train-manifest PATH --eval-results PATH --eval-summary PATH
  --log PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_DATASET_ID = (
    "prasadpattuvamveedu/voyari-tiny-day4-step25-backup-20261010"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config", {})
    if (
        checkpoint.get("stage") != "sft_day4"
        or checkpoint.get("parent_stage") != "sft_v2"
        or checkpoint.get("parent_sft_step") != 100
        or int(checkpoint.get("step", -1)) != 25
        or config.get("parent_stage") != "sft_v2"
        or config.get("parent_sft_step") != 100
        or "model_state_dict" not in checkpoint
        or "optimizer_state_dict" not in checkpoint
    ):
        raise ValueError("Expected full Day 4 step-25 checkpoint based on SFT v2 step 100")
    return {
        "stage": checkpoint["stage"], "step": checkpoint["step"],
        "parent_stage": checkpoint["parent_stage"],
        "parent_sft_step": checkpoint["parent_sft_step"],
        "parent_checkpoint_sha256": config.get("parent_checkpoint_sha256"),
        "training_data_sha256": config.get("data_sha256"),
        "tokenizer_sha256": config.get("tokenizer_sha256"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Private Kaggle backup of Day 4 step 25")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify existing private dataset without re-uploading")
    parser.add_argument("--verify-attempts", type=int, default=8)
    parser.add_argument("--verify-interval", type=float, default=8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--eval-results", type=Path)
    parser.add_argument("--eval-summary", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--backup-dir", type=Path,
        default=Path("/kaggle/working/Voyari_Day4_Step25_Private_Backup"),
    )
    return parser.parse_args()


def verify_remote(dataset_id, expected, attempts=8, interval=8):
    """Kaggle dataset creation is asynchronous; listing may temporarily fail."""
    if attempts < 1 or interval < 0:
        raise ValueError("Verification attempts must be positive and interval nonnegative")
    expected = set(expected)
    for attempt in range(1, attempts + 1):
        listed = subprocess.run(
            ["kaggle", "datasets", "files", dataset_id, "--page-size", "200"],
            capture_output=True, text=True,
        )
        missing = sorted(name for name in expected if name not in listed.stdout)
        if listed.returncode == 0 and not missing:
            print("PRIVATE DATASET BACKUP FILE LIST VERIFIED", flush=True)
            print("Verified files:", len(expected), flush=True)
            for name in sorted(expected):
                print("FOUND:", name, flush=True)
            print("Dataset URL: https://www.kaggle.com/datasets/" + dataset_id)
            print("Please confirm Dataset Visibility = Private in Kaggle UI.")
            return
        print(
            f"Remote listing attempt {attempt}/{attempts}: "
            f"exit={listed.returncode}; missing={missing}",
            flush=True,
        )
        if listed.stdout.strip():
            print("Kaggle response:", listed.stdout[-1200:], flush=True)
        if listed.stderr.strip():
            print("Kaggle error:", listed.stderr[-1200:], flush=True)
        if attempt < attempts:
            time.sleep(interval)
    raise RuntimeError(
        "Dataset creation/upload was requested but the remote files are not "
        "fully listed yet. Use --verify-only again; do NOT rerun dataset create."
    )


def main():
    args = parse_args()
    if not args.dataset_id.startswith("prasadpattuvamveedu/"):
        raise ValueError("Only an account-owned dataset ID may be used")
    if args.verify_only:
        manifest = args.backup_dir / "day4_backup_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Local backup manifest missing: {manifest}. "
                "Original upload used this directory."
            )
        backup = json.loads(manifest.read_text(encoding="utf-8"))
        if backup.get("dataset_id") != args.dataset_id:
            raise ValueError("Dataset ID differs from the backed-up manifest")
        expected = {item["name"] for item in backup["files"]}
        expected.add(manifest.name)
        verify_remote(args.dataset_id, expected, args.verify_attempts, args.verify_interval)
        return

    required = [
        args.checkpoint, args.train, args.train_manifest,
        args.eval_results, args.eval_summary, args.log,
    ]
    if any(file is None for file in required):
        raise ValueError(
            "Provide all six file paths to create a backup, "
            "or use --verify-only for the existing dataset"
        )
    for file in required:
        if not file.is_file():
            raise FileNotFoundError(file)
    if len(set(p.name for p in required)) != len(required):
        raise ValueError("Input basenames must be unique")
    if len(set(p.resolve() for p in required)) != len(required):
        raise ValueError("Input paths must be unique")
    checkpoint = verify_checkpoint(args.checkpoint)
    test_summary = json.loads(args.eval_summary.read_text(encoding="utf-8"))
    if (
        test_summary.get("stage") != "sft_day4"
        or int(test_summary.get("step", -1)) != 25
        or int(test_summary.get("total", -1)) != 30
        or test_summary.get("benchmark_sha256")
        != "df424614a58505b2853318e23d1cd85f05242e65ca6dbd8f37a0b372ef103eb4"
    ):
        raise ValueError("Evaluation summary does not match Day 4 step 25")

    folder = args.backup_dir
    folder.mkdir(parents=True, exist_ok=True)
    resolved_folder = folder.resolve()
    if any(p.resolve().parent == resolved_folder for p in required):
        raise ValueError("Backup source files must be outside the backup directory")

    files = []
    for src in required:
        dest = folder / src.name
        if not dest.is_file() or dest.stat().st_size != src.stat().st_size or sha256(dest) != sha256(src):
            shutil.copy2(src, dest)
        files.append({
            "name": src.name,
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
        })

    manifest_path = folder / "day4_backup_manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset_id": args.dataset_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "files": files,
        "note": "Private backup, separate from the immutable Day 3 SFT v2 checkpoint.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": "VoyariLM Tiny Day 4 Step 25 Private Backup",
        "id": args.dataset_id,
        "licenses": [{"name": "other"}],
        "description": (
            "Private VoyariLM Tiny SFT Day 4 checkpoint, training data "
            "and evaluation evidence. Proprietary project artifacts; "
            "not licensed for public redistribution."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    print("Checkpoint validated:", checkpoint["stage"], checkpoint["step"], flush=True)
    print("Preparing private dataset:", args.dataset_id, flush=True)
    print("No --public option is used.", flush=True)
    create = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(folder), "-t"],
        capture_output=True, text=True,
    )
    print(create.stdout[-2500:], flush=True)
    if create.returncode or "dataset creation error" in create.stdout.lower() or "dataset creation error" in create.stderr.lower():
        raise RuntimeError(
            "Kaggle dataset creation failed:\n" + create.stdout[-2500:]
            + "\n" + create.stderr[-2500:]
        )

    expected = {p["name"] for p in files} | {manifest_path.name}
    verify_remote(args.dataset_id, expected, args.verify_attempts, args.verify_interval)


if __name__ == "__main__":
    main()
