"""Regression tests for Day 4 parent, checkpoint isolation and scheduler.

Run: python scripts/test_train_sft_day4.py
No GPU, no training, no external datasets required.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from train_sft_day4 import (
    latest_checkpoint, lr_at_step, resolve_resume, validate_parent_checkpoint,
)


class Day4TrainerTests(unittest.TestCase):
    def test_approved_parent(self):
        validate_parent_checkpoint({
            "stage": "sft_v2", "step": 100,
            "run_config": {"parent_sft_step": 8000},
        })

    def test_reject_v1_parent(self):
        with self.assertRaises(RuntimeError):
            validate_parent_checkpoint({
                "stage": "sft", "step": 8000,
                "run_config": {},
            })

    def test_reject_wrong_day3_step(self):
        with self.assertRaises(RuntimeError):
            validate_parent_checkpoint({
                "stage": "sft_v2", "step": 75,
                "run_config": {"parent_sft_step": 8000},
            })

    def test_reject_wrong_parent_lineage(self):
        with self.assertRaises(RuntimeError):
            validate_parent_checkpoint({
                "stage": "sft_v2", "step": 100,
                "run_config": {"parent_sft_step": 7000},
            })

    def test_day4_prefix_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "voyari_sft_step_000100.pt").touch()
            (folder / "voyari_day4_step_000025.pt").touch()
            (folder / "voyari_day4_step_000050.pt").touch()
            self.assertEqual(
                latest_checkpoint(folder).name, "voyari_day4_step_000050.pt"
            )

    def test_auto_resume_and_none_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            checkpoint = folder / "voyari_day4_step_000025.pt"
            checkpoint.touch()
            self.assertEqual(resolve_resume("auto", folder), checkpoint)
            with self.assertRaises(RuntimeError):
                resolve_resume("none", folder)

    def test_learning_rate_stays_bounded(self):
        for step in range(1, 101):
            lr = lr_at_step(step, 100, 8, 3e-6, 8e-7)
            self.assertGreater(lr, 0)
            self.assertLessEqual(lr, 3e-6)


if __name__ == "__main__":
    unittest.main()
