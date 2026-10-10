"""Offline unit checks for the Day 4 diagnostic scoring rules."""
from __future__ import annotations
import unittest
from day4_diagnostic import evaluate_answer


class DiagnosticScoreTests(unittest.TestCase):
    def test_state_exact(self):
        case = {
            "id": "test_state", "category": "state", "question": "test",
            "expected": {"type": "state_update", "state": {"origin": "Mysuru", "duration_days": 4}},
        }
        self.assertTrue(evaluate_answer(
            case, '{"type":"state_update","state":{"origin":"Mysuru","duration_days":4}}'
        )["exact_match"])

    def test_state_extra_key_fails(self):
        case = {
            "id": "test_state_extra", "category": "state", "question": "test",
            "expected": {"type": "state_update", "state": {"origin": "Mysuru", "duration_days": 4}},
        }
        result = evaluate_answer(
            case,
            '{"type":"state_update","state":{"origin":"Mysuru","duration_days":4,"traveller_count":4}}',
        )
        self.assertTrue(result["required_fields_match"])
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["unexpected_fields"], ["traveller_count"])

    def test_state_wrong_duration_fails(self):
        case = {
            "id": "wrong_duration", "category": "state", "question": "test",
            "expected": {"type": "state_update", "state": {"duration_days": 5}},
        }
        self.assertFalse(evaluate_answer(
            case, '{"type":"state_update","state":{"duration_days":7}}'
        )["exact_match"])

    def test_tool_wrong_location_fails(self):
        case = {
            "id": "wrong_location", "category": "tool", "question": "test",
            "expected": {"type": "tool_call", "tool": "weather",
                         "arguments": {"location": "Nashik", "date_expression": "tomorrow"}},
        }
        self.assertFalse(evaluate_answer(
            case, '{"type":"tool_call","tool":"weather",'
                  '"arguments":{"location":"Rishikesh","date_expression":"tomorrow"}}'
        )["exact_match"])

    def test_clarification_must_address_topic(self):
        case = {
            "id": "missing_dates", "category": "clarify", "question": "test",
            "expected": {"type": "message"}, "checks": {"topic_any": ["date", "when"]},
        }
        self.assertFalse(evaluate_answer(
            case, '{"type":"message","message":"Which city?"}'
        )["exact_match"])
        self.assertTrue(evaluate_answer(
            case, '{"type":"message","message":"Which dates do you have in mind?"}'
        )["exact_match"])

    def test_invalid_json_fails(self):
        case = {
            "id": "invalid_json", "category": "state", "question": "test",
            "expected": {"type": "state_update", "state": {"duration_days": 3}},
        }
        self.assertFalse(evaluate_answer(case, '{"type":')['valid_json'])


if __name__ == "__main__":
    unittest.main()
