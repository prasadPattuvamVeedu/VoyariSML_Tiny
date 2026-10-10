"""Unit tests for Day 4 Kaggle backup verification and delayed indexing."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import backup_day4_checkpoint as backup


class Day4BackupVerificationTests(unittest.TestCase):
    def test_delayed_indexing_then_success(self):
        first = subprocess.CompletedProcess(args=[], returncode=1, stdout="Not ready", stderr="")
        second = subprocess.CompletedProcess(args=[], returncode=0, stdout=(
            "name size\n"
            "voyari_day4_step_000025.pt 354MB\n"
            "day4_backup_manifest.json 2KB\n"
        ), stderr="")
        with patch.object(backup.subprocess, "run", side_effect=[first, second]) as run, \
             patch.object(backup.time, "sleep") as snooze, \
             contextlib.redirect_stdout(io.StringIO()) as out:
            backup.verify_remote(
                backup.DEFAULT_DATASET_ID,
                ["voyari_day4_step_000025.pt", "day4_backup_manifest.json"],
                attempts=3, interval=1,
            )
        self.assertEqual(run.call_count, 2)
        snooze.assert_called_once_with(1)
        self.assertIn("PRIVATE DATASET BACKUP FILE LIST VERIFIED", out.getvalue())
        self.assertTrue(all("create" not in call.args[0] for call in run.call_args_list))

    def test_incomplete_listing_not_success(self):
        reply = subprocess.CompletedProcess(args=[], returncode=0, stdout="day4_backup_manifest.json", stderr="")
        with patch.object(backup.subprocess, "run", return_value=reply), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "not fully listed"):
                backup.verify_remote(
                    backup.DEFAULT_DATASET_ID, ["voyari_day4_step_000025.pt"],
                    attempts=1, interval=0,
                )

    def test_verify_only_never_reuploads(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "day4_backup_manifest.json").write_text(json.dumps({
                "dataset_id": backup.DEFAULT_DATASET_ID,
                "files": [{"name": "voyari_day4_step_000025.pt"}],
            }), encoding="utf-8")
            with patch.object(sys, "argv", [
                "backup_day4_checkpoint.py", "--verify-only", "--backup-dir", root,
                "--verify-attempts", "1", "--verify-interval", "0",
            ]), patch.object(backup, "verify_remote") as verify, \
                 patch.object(backup.subprocess, "run") as run:
                backup.main()
            verify.assert_called_once_with(
                backup.DEFAULT_DATASET_ID,
                {"voyari_day4_step_000025.pt", "day4_backup_manifest.json"},
                1, 0.0,
            )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
