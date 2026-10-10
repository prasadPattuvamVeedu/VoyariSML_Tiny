"""No-GPU regression tests for Day 4 evaluation comparison."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare_day4_checkpoints import EXPECTED_BENCHMARK_SHA, load

SCRIPT = Path(__file__).with_name("compare_day4_checkpoints.py")


class ComparisonTests(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "dupes.jsonl"
            entry = {"id": "d4_01", "question": "Q"}
            file.write_text(json.dumps(entry) + "\n" + json.dumps(entry) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate case"):
                load(file)

    def test_30_case_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            old = directory / "old.jsonl"
            new = directory / "new.jsonl"
            old_summary = directory / "old_summary.json"
            new_summary = directory / "new_summary.json"
            old_rows = []
            new_rows = []
            for i in range(30):
                category = ("state" if i < 12 else "tool" if i < 20 else "clarify")
                expected = {"type": ("state_update" if category == "state"
                                     else "tool_call" if category == "tool" else "message")}
                shared = {
                    "id": f"d4_{i+1:02d}", "category": category,
                    "question": f"Example {i}", "expected": expected, "valid_json": True,
                }
                old_rows.append({
                    **shared, "action_match": True, "exact_match": category == "clarify",
                    "predicted": {"type": expected["type"], "message": "Which dates?"},
                    "topic_match": category == "clarify",
                })
                new_rows.append({
                    **shared, "action_match": category != "clarify", "exact_match": False,
                    "predicted": {"type": "state_update"},
                    "topic_match": False,
                })
            for path, rows in ((old, old_rows), (new, new_rows)):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            for path, stage, step in ((old_summary, "sft_v2", 100),
                                       (new_summary, "sft_day4", 25)):
                categories = {
                    name: {
                        "total": amount,
                        "action_match": amount if stage == "sft_v2" or name != "clarify" else 0,
                        "exact_or_topic_match": (
                            amount if stage == "sft_v2" and name == "clarify" else 0
                        ),
                    }
                    for name, amount in (("state", 12), ("tool", 8), ("clarify", 10))
                }
                path.write_text(json.dumps({
                    "stage": stage, "step": step, "total": 30,
                    "benchmark_sha256": EXPECTED_BENCHMARK_SHA,
                    "categories": categories,
                }), encoding="utf-8")
            process = subprocess.run([
                sys.executable, str(SCRIPT),
                "--baseline", str(old), "--candidate", str(new),
                "--baseline-summary", str(old_summary),
                "--candidate-summary", str(new_summary),
                "--category", "clarify", "--limit", "2",
                "--report", str(directory / "comparison.json"),
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("DAY 4 — MODEL REGRESSION REVIEW", process.stdout)
            self.assertIn("clarify: action 10/10 -> 0/10", process.stdout)
            report = json.loads((directory / "comparison.json").read_text())
            self.assertEqual(len(report["cases"]), 30)
            self.assertEqual(
                next(
                    t["count"] for t in report["category_action_transitions"]
                    if t["category"] == "clarify" and t["outcome"] == "action_regressed"
                ), 10
            )


if __name__ == "__main__":
    unittest.main()
