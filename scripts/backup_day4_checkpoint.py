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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--backup-dir", type=Path,
        default=Path("/kaggle/working/Voyari_Day4_Step25_Private_Backup"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    required = [
        args.checkpoint, args.train, args.train_manifest,
        args.eval_results, args.eval_summary, args.log,
    ]
    for file in required:
        if not file.is_file():
            raise FileNotFoundError(file)
    if len(set(p.name for p in required)) != len(required):
        raise ValueError("Input basenames must be unique")
    if len(set(p.resolve() for p in required)) != len(required):
        raise ValueError("Input paths must be unique")
    if not args.dataset_id.startswith("prasadpattuvamveedu/"):
        raise ValueError("Only an account-owned dataset ID may be used")

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

    listed = subprocess.run(
        ["kaggle", "datasets", "files", args.dataset_id, "--page-size", "200"],
        capture_output=True, text=True,
    )
    if listed.returncode:
        raise RuntimeError("Remote verification failed: " + listed.stderr[-2000:])
    expected = {p["name"] for p in files} | {
        "day4_backup_manifest.json",
    }
    missing = sorted(name for name in expected if name not in listed.stdout)
    if missing:
        raise RuntimeError(
            "Remote dataset created but files NOT confirmed: " + repr(missing)
            + "\nDo not close Kaggle until resolved."
        )
    print("PRIVATE DATASET BACKUP FILE LIST VERIFIED")
    print("Verified files:", len(expected))
    for name in sorted(expected):
        print("FOUND:", name)
    print("Dataset URL: https://www.kaggle.com/datasets/" + args.dataset_id)
    print(
        "Check the dataset's Visibility in Kaggle UI. "
        "The upload did not specify --public."
    )


if __name__ == "__main__":
    main()
