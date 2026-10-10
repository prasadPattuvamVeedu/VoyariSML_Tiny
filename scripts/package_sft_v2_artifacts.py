"""Package reviewed SFT v2 data and record checkpoint checksums for safe backup.

Read-only for input data/checkpoint; writes a ZIP and JSON manifest in output-dir.
This does not upload files or guarantee that Kaggle session outputs are persistent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import torch

from audit_sft_v2_data import inspect_action, iter_records, normalize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_reviewed(path: Path):
    conversations = 0
    turns = 0
    types = Counter()
    warnings = Counter()
    for _, record in iter_records(path):
        conversations += 1
        history = []
        for message in record.get("messages", []):
            role, content = message.get("role"), message.get("content", "")
            if role == "user":
                history.append(normalize(content))
            elif role == "assistant":
                turns += 1
                action = json.loads(content) if isinstance(content, str) else content
                if not isinstance(action, dict):
                    raise ValueError("Assistant action is not an object")
                types[action.get("type", "unknown")] += 1
                warnings.update(inspect_action(action, " ".join(history)))
    if (conversations, turns) != (960, 1447):
        raise ValueError(f"Unexpected dataset size: {conversations}, {turns}")
    expected = {"tool_call": 339, "message": 816, "state_update": 292}
    if dict(types) != expected:
        raise ValueError(f"Unexpected action distribution: {dict(types)}")
    if warnings:
        raise ValueError(f"Reviewed training file still has audit warnings: {dict(warnings)}")
    return {"conversations": conversations, "assistant_turns": turns, "action_types": dict(types)}


def main():
    parser = argparse.ArgumentParser(description="Package reviewed SFT v2 data backup")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reviewed = args.data_dir / "mixed_train_reviewed_candidate.jsonl"
    if not reviewed.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Reviewed dataset or SFT v2 step-100 checkpoint missing")

    data_stats = audit_reviewed(reviewed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "sft_v2" or int(checkpoint.get("step", -1)) != 100:
        raise ValueError("Expected an SFT v2 step-100 checkpoint")
    if checkpoint.get("run_config", {}).get("parent_sft_step") != 8000:
        raise ValueError("Expected SFT v1 8000 parent checkpoint")
    del checkpoint

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / "Voyari_SFT_v2_Reviewed_Data.zip"
    manifest_path = args.output_dir / "Voyari_SFT_v2_Reviewed_Manifest.json"
    if zip_path.resolve() == args.checkpoint.resolve():
        raise ValueError("Backup ZIP must not overwrite model checkpoint")

    files = sorted(
        p for p in args.data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".txt", ".md")
    )
    if not files:
        raise ValueError("No data JSON/JSONL files found")
    manifest = {
        "reviewed_data": data_stats,
        "checkpoint": {
            "path_in_current_kaggle_session": str(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
            "included_in_zip": False,
            "important": "Checkpoint MUST be saved separately to persistent storage.",
        },
        "included_data_files": [],
        "note": (
            "Files are backed up locally only until downloaded or saved in a "
            "persistent private Kaggle Dataset. Never commit model weights or "
            "private training data to GitHub."
        ),
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            archive_name = str(path.relative_to(args.data_dir)).replace("\\", "/")
            manifest["included_data_files"].append({
                "name": archive_name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
            bundle.write(path, arcname=archive_name)

    with zipfile.ZipFile(zip_path) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        if len(bundle.namelist()) != len(files):
            raise RuntimeError("ZIP member count mismatch")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("VOYARILM SFT V2 BACKUP PREPARED")
    print("Reviewed conversations:", data_stats["conversations"])
    print("Reviewed assistant turns:", data_stats["assistant_turns"])
    print("Files in data ZIP:", len(files))
    print("Data ZIP:", zip_path)
    print("Data ZIP MB:", round(zip_path.stat().st_size / 1024 ** 2, 2))
    print("Manifest:", manifest_path)
    print("Checkpoint:", args.checkpoint)
    print("Checkpoint MB:", round(args.checkpoint.stat().st_size / 1024 ** 2, 2))
    print("IMPORTANT: Save checkpoint separately; local output is not persistent.")


if __name__ == "__main__":
    main()
