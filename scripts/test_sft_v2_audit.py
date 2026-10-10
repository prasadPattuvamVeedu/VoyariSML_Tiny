"""Regression checks for SFT audit heuristics. Run: python scripts/test_sft_v2_audit.py"""
from __future__ import annotations

import unittest

from audit_sft_v2_data import (
    inspect_action, duration_mentioned, budget_supported, first_eval_question,
)


def state(group, count):
    return {
        "type": "state_update",
        "state": {"traveller_group": group, "traveller_count": count},
    }


class AuditHeuristicTests(unittest.TestCase):
    def test_behavior_eval_question_extraction(self):
        self.assertEqual(
            first_eval_question({"question": "  Visit Kodaikanal? ", "expected": {}}),
            "visit kodaikanal?",
        )

    def test_conversation_eval_question_extraction(self):
        self.assertEqual(
            first_eval_question({"messages": [
                {"role": "system", "content": "Travel"},
                {"role": "user", "content": "  Visit Ooty? "},
            ]}),
            "visit ooty?",
        )

    def test_solo_means_one(self):
        flags = inspect_action(state("solo", 1), "jaipur start, 2 days, solo")
        self.assertNotIn("traveller_count_not_numeric_in_user_context", flags)

    def test_solo_plural_requires_review(self):
        flags = inspect_action(
            state("solo", 1), "3 days from chennai, solo. 15000 total for all of us"
        )
        self.assertIn("solo_with_plural_reference_review", flags)
        self.assertNotIn("traveller_count_not_numeric_in_user_context", flags)

    def test_solo_with_our_budget_requires_review(self):
        flags = inspect_action(
            state("solo", 1), "2 days from kochi, solo. our full budget is 40000"
        )
        self.assertIn("solo_with_plural_reference_review", flags)

    def test_group_default_four_flagged(self):
        flags = inspect_action(
            state("friends", 4), "5 days from jaipur, with friends. budget 20000"
        )
        self.assertIn("traveller_count_not_numeric_in_user_context", flags)

    def test_explicit_family_size_preserved(self):
        flags = inspect_action(
            state("family", 4), "we are a family of four from delhi"
        )
        self.assertNotIn("traveller_count_not_numeric_in_user_context", flags)

    def test_partner_implies_two(self):
        flags = inspect_action(
            state("couple", 2), "travelling with my partner"
        )
        self.assertNotIn("traveller_count_not_numeric_in_user_context", flags)

    def test_missing_hotel_date_flagged(self):
        action = {
            "type": "tool_call", "tool": "accommodation_search",
            "arguments": {"location": "Ooty", "date_expression": "next_weekend"},
        }
        self.assertIn(
            "tool_date_without_explicit_date",
            inspect_action(action, "find current hotel availability in ooty"),
        )

    def test_given_next_friday_date_not_missing(self):
        action = {
            "type": "tool_call", "tool": "weather",
            "arguments": {"location": "Alleppey", "date_expression": "next Friday"},
        }
        self.assertNotIn(
            "tool_date_without_explicit_date",
            inspect_action(action, "weather in alleppey next friday"),
        )

    def test_multiturn_day_arithmetic(self):
        self.assertTrue(duration_mentioned(
            4, "i have 3 days. add one more day."
        ))

    def test_multiturn_budget_arithmetic(self):
        self.assertTrue(budget_supported(
            10000, "budget of ₹15,000. reduce the budget by ₹5,000."
        ))


if __name__ == "__main__":
    unittest.main()
