"""Fast CPU-only Day 5 generator and schema tests. Run: python scripts/test_day5.py"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from day5_schema import SYSTEM_TEXT, validate_action
from generate_day5_training import state_case, tool_case, clarification_case, multi_case
from validate_day5_training import validate


class Day5Tests(unittest.TestCase):
    def test_schema_regression(self):
        q = "I want to visit Jaipur for 3 days with a budget of ₹5,000."
        good = {"type": "state_update", "state": {"destination": "Jaipur", "duration_days": 3, "budget_inr": 5000}}
        self.assertEqual(validate_action(good, q), [])
        bad = {"type": "state_update", "state": {"origin": "Jaipur", "duration_days": 3, "budget_inr": 8000,
                                                   "traveller_group": "friends", "interests": ["food"]}}
        self.assertTrue(validate_action(bad, q))
        wrongly_swapped = {"type": "state_update", "state": {"origin": "Jaipur"}}
        self.assertIn("origin_role_cue_missing", validate_action(wrongly_swapped, q))

    def test_tool_regression(self):
        a = {"type": "tool_call", "tool": "weather", "arguments": {"location": "Pune", "date_expression": "today"}}
        self.assertEqual(validate_action(a, "Check the weather in Pune today."), [])
        a["arguments"]["location"] = "Jaipur"
        self.assertTrue(validate_action(a, "Check the weather in Pune today."))

    def test_generator_samples(self):
        rng = random.Random(20261011)
        for fn in (state_case, tool_case, clarification_case, multi_case):
            for _ in range(300):
                record = fn(rng, SYSTEM_TEXT)
                user = ""
                for message in record["messages"]:
                    if message["role"] == "user":
                        user = message["content"]
                    elif message["role"] == "assistant":
                        self.assertEqual(validate_action(json.loads(message["content"]), user), [])

    def test_validator_rejects_mistakes(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "bad.jsonl"
            record = {"messages": [{"role": "system", "content": SYSTEM_TEXT},
                        {"role": "user", "content": "Visit Jaipur with ₹5,000"},
                        {"role": "assistant", "content": json.dumps({"type": "state_update", "state": {"budget_inr": 8000}})}]}
            file.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
