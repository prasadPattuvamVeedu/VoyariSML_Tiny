"""Regression tests for grounded Day 4 SFT generator.

Run: python scripts/test_day4_grounded_generation.py
No GPU or external dataset required.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_day4_grounded_sft.py")


class GeneratedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.system_file = root / "system.jsonl"
        self.system_file.write_text(
            json.dumps({"messages": [
                {"role": "system", "content": "Respond using travel JSON."},
                {"role": "user", "content": "Baseline example"},
                {"role": "assistant", "content": '{"type":"message","message":"Hello"}'},
            ]}) + "\n", encoding="utf-8"
        )
        self.output = root / "generated.jsonl"
        self.manifest = root / "manifest.json"

    def run_generator(self, *extra):
        # System file is intentionally repeated in the overlap checks.
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--system-dataset", str(self.system_file),
                "--existing-train", str(self.system_file),
                "--output", str(self.output),
                "--manifest", str(self.manifest),
                *extra,
            ],
            capture_output=True, text=True, check=False,
        )

    def test_duplicate_input_roles_allowed(self):
        result = self.run_generator("--state", "5", "--clarify", "0", "--tool", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(x) for x in self.output.read_text(
            encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(records), 5)
        self.assertTrue(all(
            json.loads(r["messages"][-1]["content"])["type"] == "state_update"
            for r in records
        ))

    def test_cannot_overwrite_system_source(self):
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--system-dataset", str(self.system_file),
                "--existing-train", str(self.system_file),
                "--output", str(self.system_file),
                "--manifest", str(self.manifest),
                "--state", "1", "--clarify", "0", "--tool", "0",
            ],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not overwrite any input files", result.stderr)

    def test_can_generate_240_unique_clarifications(self):
        result = self.run_generator(
            "--state", "0", "--clarify", "240", "--tool", "0",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(x) for x in self.output.read_text(
            encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(records), 240)
        prompts = [r["messages"][1]["content"] for r in records]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertTrue(all(
            json.loads(r["messages"][-1]["content"])["type"] == "message"
            for r in records
        ))


if __name__ == "__main__":
    unittest.main()
