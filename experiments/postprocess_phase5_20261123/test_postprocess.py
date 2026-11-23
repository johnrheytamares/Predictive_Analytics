"""Focused Phase 5 post-processing tests."""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from experiment import EXPECTED_DAILY_SHA, PHASE2_DAILY, load_demand, load_predictions, verify_against_demand
from postprocess import apply_variant_a, apply_variant_b, guest_floor_changes


class VerificationTests(unittest.TestCase):
    def test_phase2_sha(self):
        digest = hashlib.sha256(PHASE2_DAILY.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_DAILY_SHA)

    def test_prediction_actuals_match_demand(self):
        verify_against_demand(load_predictions(), load_demand())


class PostprocessRuleTests(unittest.TestCase):
    def test_variant_a_applies_party_floor(self):
        bookings, guests = apply_variant_a([1.2], [2.4], party_size=7.0)
        self.assertEqual(bookings, [1])
        self.assertEqual(guests, [7])  # max(2, 7, 1)

    def test_variant_b_skips_party_floor(self):
        bookings, guests = apply_variant_b([1.2], [2.4])
        self.assertEqual(bookings, [1])
        self.assertEqual(guests, [2])

    def test_positive_guests_force_booking(self):
        a_b, a_g = apply_variant_a([0.1], [5.0], party_size=7.0)
        b_b, b_g = apply_variant_b([0.1], [5.0])
        self.assertEqual(a_b[0], 1)
        self.assertEqual(b_b[0], 1)
        self.assertGreaterEqual(a_g[0], a_b[0])
        self.assertGreaterEqual(b_g[0], b_b[0])

    def test_guest_floor_change_detection(self):
        impact = guest_floor_changes([2.0], [3.0], party_size=7.0)
        self.assertEqual(impact["changed_days"], 1)
        self.assertEqual(impact["guest_total_increase"], 11)  # 14 vs 3


if __name__ == "__main__":
    unittest.main()
