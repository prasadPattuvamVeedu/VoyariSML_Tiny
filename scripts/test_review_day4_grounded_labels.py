"""Small unit tests for the Day 4 independent label reviewer (no training)."""
from __future__ import annotations

import unittest
from review_day4_grounded_labels import warnings_for


class LabelGroundingTests(unittest.TestCase):
    def test_correct_budget_and_origin(self):
        question = ("My starting city is Kochi. I have five days. "
                    "My per-person budget is ₹12,500. I prefer wildlife.")
        action = {"type": "state_update", "state": {
            "origin": "Kochi", "duration_days": 5, "budget_inr": 12500,
            "budget_basis": "per_person", "interests": ["wildlife"],
        }}
        self.assertEqual(warnings_for(question, action), [])

    def test_wrong_budget_basis(self):
        question = "I'm leaving from Delhi. I have 3 days. The budget is ₹9,000 per person."
        action = {"type": "state_update", "state": {
            "origin": "Delhi", "duration_days": 3,
            "budget_inr": 9000, "budget_basis": "total",
        }}
        self.assertIn("total_budget_not_grounded", warnings_for(question, action))

    def test_invented_state_values(self):
        question = "My starting city is Patna. I have 4 days."
        action = {"type": "state_update", "state": {
            "origin": "Chennai", "duration_days": 7,
            "budget_inr": 17000, "budget_basis": "total",
        }}
        issues = warnings_for(question, action)
        self.assertIn("origin_not_grounded", issues)
        self.assertIn("duration_not_grounded", issues)
        self.assertIn("budget_not_literal", issues)

    def test_group_count_without_evidence(self):
        question = "I'm leaving from Kochi. I'll travel for 3 days. I'm travelling with friends."
        action = {"type": "state_update", "state": {
            "origin": "Kochi", "duration_days": 3,
            "traveller_group": "friends", "traveller_count": 4,
        }}
        self.assertIn("assumed_unknown_group_size", warnings_for(question, action))

    def test_weather_arguments_grounded(self):
        question = "Please check the weather in Ranchi tomorrow."
        action = {"type": "tool_call", "tool": "weather", "arguments": {
            "location": "Ranchi", "date_expression": "tomorrow",
        }}
        self.assertEqual(warnings_for(question, action), [])

    def test_unsupported_weather_location(self):
        question = "Please check the weather in Delhi tomorrow."
        action = {"type": "tool_call", "tool": "weather", "arguments": {
            "location": "Hyderabad", "date_expression": "tomorrow",
        }}
        self.assertIn("weather_location_not_grounded", warnings_for(question, action))

    def test_clarify_missing_dates(self):
        question = "Find hotel rooms in Nagpur but I don't know my dates."
        good = {"type": "message", "message": "What are your check-in and check-out dates?"}
        bad = {"type": "message", "message": "Which city are you travelling to?"}
        self.assertEqual(warnings_for(question, good), [])
        self.assertIn("hotel_dates_not_requested", warnings_for(question, bad))


if __name__ == "__main__":
    unittest.main()
