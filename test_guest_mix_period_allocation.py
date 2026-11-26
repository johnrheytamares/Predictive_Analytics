"""Local tests for aggregate-preserving guest-category allocation.

Does not train, deploy, or touch the database.
"""
from __future__ import annotations

import unittest

from predict_real import (
    GUEST_MIX_CATEGORIES,
    allocate_guest_mix_period,
    apply_period_guest_mix,
    estimate_guest_mix,
)

# Frozen v3.1 proportions from forecast_models_metadata.json
V31_METADATA = {
    "guest_mix_ready": True,
    "guest_mix_proportions": {
        "adults": 0.69725596,
        "children": 0.19358225,
        "seniors": 0.06537712,
        "infants": 0.03853651,
        "unknown_guests": 0.00524816,
    },
}


def _sum_mix(rows):
    return {key: sum(row[key] for row in rows) for key in GUEST_MIX_CATEGORIES}


class GuestMixPeriodAllocationTests(unittest.TestCase):
    def test_period_total_thirteen_matches_single_shot(self):
        desired = estimate_guest_mix(13, V31_METADATA)
        self.assertEqual(
            desired,
            {
                "adults": 9,
                "children": 3,
                "seniors": 1,
                "infants": 0,
                "unknown_guests": 0,
            },
        )

    def test_thirteen_guest_sparse_days_no_longer_all_adults(self):
        """Days of 1–2 guests that previously summed to 13 adults."""
        daily = [2, 2, 2, 2, 2, 2, 1]
        self.assertEqual(sum(daily), 13)
        naive = [estimate_guest_mix(g, V31_METADATA) for g in daily]
        self.assertEqual(_sum_mix(naive)["adults"], 13)
        self.assertEqual(_sum_mix(naive)["children"], 0)

        allocated = allocate_guest_mix_period(daily, V31_METADATA)
        self.assertEqual(
            _sum_mix(allocated),
            {
                "adults": 9,
                "children": 3,
                "seniors": 1,
                "infants": 0,
                "unknown_guests": 0,
            },
        )
        for day_total, row in zip(daily, allocated):
            self.assertEqual(sum(row.values()), day_total)

    def test_thirteen_ones(self):
        daily = [1] * 13
        allocated = allocate_guest_mix_period(daily, V31_METADATA)
        self.assertEqual(
            _sum_mix(allocated),
            {
                "adults": 9,
                "children": 3,
                "seniors": 1,
                "infants": 0,
                "unknown_guests": 0,
            },
        )
        self.assertTrue(all(sum(row.values()) == 1 for row in allocated))

    def test_zero_guests_period(self):
        daily = [0, 0, 0, 0]
        allocated = allocate_guest_mix_period(daily, V31_METADATA)
        self.assertEqual(len(allocated), 4)
        self.assertEqual(_sum_mix(allocated), {key: 0 for key in GUEST_MIX_CATEGORIES})

    def test_mixed_zeros_and_sparse(self):
        daily = [0, 1, 0, 2, 0, 1, 0]
        allocated = allocate_guest_mix_period(daily, V31_METADATA)
        desired = estimate_guest_mix(sum(daily), V31_METADATA)
        self.assertEqual(_sum_mix(allocated), desired)
        for day_total, row in zip(daily, allocated):
            self.assertEqual(sum(row.values()), day_total)
            if day_total == 0:
                self.assertEqual(row, {key: 0 for key in GUEST_MIX_CATEGORIES})

    def test_seven_thirty_ninety_day_lengths(self):
        for days in (7, 30, 90):
            # Sparse repeating pattern; totals stay modest.
            daily = [((i % 5) + 1) if i % 3 else 0 for i in range(days)]
            allocated = allocate_guest_mix_period(daily, V31_METADATA)
            desired = estimate_guest_mix(sum(daily), V31_METADATA)
            self.assertEqual(len(allocated), days)
            self.assertEqual(_sum_mix(allocated), desired)
            for day_total, row in zip(daily, allocated):
                self.assertEqual(sum(row.values()), day_total)

    def test_apply_period_guest_mix_preserves_bookings_and_guests(self):
        forecasts = [
            {"predicted_bookings": 1, "predicted_total_guests": 2},
            {"predicted_bookings": 1, "predicted_total_guests": 2},
            {"predicted_bookings": 1, "predicted_total_guests": 2},
            {"predicted_bookings": 1, "predicted_total_guests": 2},
            {"predicted_bookings": 1, "predicted_total_guests": 2},
            {"predicted_bookings": 1, "predicted_total_guests": 2},
            {"predicted_bookings": 0, "predicted_total_guests": 1},
        ]
        bookings_before = [row["predicted_bookings"] for row in forecasts]
        guests_before = [row["predicted_total_guests"] for row in forecasts]
        apply_period_guest_mix(forecasts, V31_METADATA)
        self.assertEqual([row["predicted_bookings"] for row in forecasts], bookings_before)
        self.assertEqual([row["predicted_total_guests"] for row in forecasts], guests_before)
        mixes = [row["estimated_guest_mix"] for row in forecasts]
        self.assertEqual(
            _sum_mix(mixes),
            {
                "adults": 9,
                "children": 3,
                "seniors": 1,
                "infants": 0,
                "unknown_guests": 0,
            },
        )

    def test_not_ready_returns_none(self):
        meta = {"guest_mix_ready": False, "guest_mix_proportions": V31_METADATA["guest_mix_proportions"]}
        self.assertIsNone(allocate_guest_mix_period([1, 2, 3], meta))


if __name__ == "__main__":
    unittest.main()
