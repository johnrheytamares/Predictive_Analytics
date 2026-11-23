"""Leakage and reproducibility checks for Phase 3 models."""
from __future__ import annotations

import hashlib
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from experiment import EXPECTED_DAILY_SHA, PHASE2_PREPARED, load_frame, serving_hashes
from models import baseline_pair, random_forest_forecast, ridge_forecast

CUTOFF = date(2026, 4, 6)


class DatasetGuardTests(unittest.TestCase):
    def test_phase2_sha(self):
        path = PHASE2_PREPARED / "data" / "daily_demand.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_DAILY_SHA)

    def test_load_frame_totals(self):
        frame = load_frame()
        self.assertEqual(len(frame), 356)
        self.assertEqual(int(frame.bookings.sum()), 519)
        self.assertEqual(int(frame.guests.sum()), 6669)


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_frame()

    def _poison_future(self, frame: pd.DataFrame) -> pd.DataFrame:
        poisoned = frame.copy()
        future = poisoned["demand_date"] >= pd.Timestamp(CUTOFF)
        poisoned.loc[future, "bookings"] = 999
        poisoned.loc[future, "guests"] = 9999
        return poisoned

    def test_baseline_ignores_future_actuals(self):
        clean = baseline_pair(self.frame, CUTOFF)
        poisoned = baseline_pair(self._poison_future(self.frame), CUTOFF)
        self.assertEqual(clean, poisoned)

    def test_ridge_ignores_future_actuals(self):
        clean = ridge_forecast(self.frame, CUTOFF, alpha_bookings=1.0, alpha_guests=1.0)
        poisoned = ridge_forecast(
            self._poison_future(self.frame), CUTOFF, alpha_bookings=1.0, alpha_guests=1.0
        )
        for target in ("bookings", "guests"):
            self.assertEqual(clean[target], poisoned[target])

    def test_random_forest_ignores_future_actuals(self):
        clean = random_forest_forecast(self.frame, CUTOFF)
        poisoned = random_forest_forecast(self._poison_future(self.frame), CUTOFF)
        for target in ("bookings", "guests"):
            self.assertEqual(clean[target], poisoned[target])

    def test_ridge_reproducible(self):
        a = ridge_forecast(self.frame, CUTOFF, alpha_bookings=10.0, alpha_guests=10.0)
        b = ridge_forecast(self.frame, CUTOFF, alpha_bookings=10.0, alpha_guests=10.0)
        self.assertEqual(a, b)

    def test_random_forest_reproducible(self):
        a = random_forest_forecast(self.frame, CUTOFF)
        b = random_forest_forecast(self.frame, CUTOFF)
        self.assertEqual(a, b)


class ServingGuardTests(unittest.TestCase):
    def test_serving_hashes_readable(self):
        hashes = serving_hashes()
        self.assertIn("forecast_models_metadata.json", hashes)
        self.assertEqual(len(hashes["forecast_models_metadata.json"]), 64)


if __name__ == "__main__":
    unittest.main()
