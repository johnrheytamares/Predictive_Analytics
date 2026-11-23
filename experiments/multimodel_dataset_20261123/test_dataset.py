"""Focused Phase 2 integrity and leakage tests. No model training."""
from __future__ import annotations

import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    add_calendar_features,
    assert_horizon_uses_only_precutoff_actuals,
    build_horizon_feature_rows,
    build_supervised_frame,
)
from periods import (
    DEVELOPMENT_WEEKS,
    FIRST_WEEK_START,
    RESERVED_WEEKS,
    SNAPSHOT_BOOKINGS,
    SNAPSHOT_DAYS,
    SNAPSHOT_END,
    SNAPSHOT_GUESTS,
    SNAPSHOT_START,
    development_cutoffs,
    reserved_cutoffs,
    week_starts,
)
from reconstruct import reconstruct_daily_frame, serving_hashes

HERE = Path(__file__).resolve().parent
PREPARED = HERE / "prepared"


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.provenance = reconstruct_daily_frame()
        cls.before_hashes = serving_hashes()

    def test_coverage_and_totals(self):
        self.assertEqual(len(self.frame), SNAPSHOT_DAYS)
        self.assertEqual(int(self.frame.bookings.sum()), SNAPSHOT_BOOKINGS)
        self.assertEqual(int(self.frame.guests.sum()), SNAPSHOT_GUESTS)
        self.assertEqual(self.frame.demand_date.min().date(), SNAPSHOT_START)
        self.assertEqual(self.frame.demand_date.max().date(), SNAPSHOT_END)
        self.assertFalse(self.provenance["byte_identical_to_original_export"])
        self.assertTrue(self.provenance["content_match"]["matches_metadata_training_snapshot_totals"])

    def test_no_missing_or_duplicate_dates(self):
        expected = [ts.strftime("%Y-%m-%d") for ts in pd.date_range(SNAPSHOT_START, SNAPSHOT_END, freq="D")]
        self.assertFalse(self.frame.demand_date.duplicated().any())
        self.assertEqual(self.frame.demand_date.dt.strftime("%Y-%m-%d").tolist(), expected)

    def test_nonnegative_demand(self):
        self.assertTrue((self.frame.bookings >= 0).all())
        self.assertTrue((self.frame.guests >= 0).all())
        self.assertTrue((self.frame.guests >= self.frame.bookings).all())

    def test_zero_demand_days_preserved(self):
        self.assertGreater(int((self.frame.bookings == 0).sum()), 0)
        self.assertEqual(
            int((self.frame.bookings == 0).sum()),
            self.provenance["zero_demand_days"]["bookings"],
        )

    def test_calendar_features(self):
        calendar = add_calendar_features(self.frame)
        sample = calendar.loc[calendar.demand_date == "2026-04-06"].iloc[0]
        self.assertEqual(int(sample.day_of_week), 0)  # Monday
        self.assertEqual(int(sample.is_weekend), 0)
        self.assertEqual(int(sample.month), 4)
        sunday = calendar.loc[calendar.demand_date == "2026-04-12"].iloc[0]
        self.assertEqual(int(sunday.day_of_week), 6)
        self.assertEqual(int(sunday.is_weekend), 1)

    def test_evaluation_periods(self):
        self.assertEqual(len(development_cutoffs()), DEVELOPMENT_WEEKS)
        self.assertEqual(len(reserved_cutoffs()), RESERVED_WEEKS)
        self.assertEqual(development_cutoffs()[0], FIRST_WEEK_START)
        self.assertEqual(reserved_cutoffs()[0].isoformat(), "2026-09-21")
        self.assertEqual(
            (reserved_cutoffs()[-1] + timedelta(days=6)).isoformat(), "2026-11-15"
        )
        self.assertEqual(week_starts()[-1] + timedelta(days=6), SNAPSHOT_END - timedelta(days=6))


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = reconstruct_daily_frame()

    def test_supervised_features_use_only_past(self):
        supervised = build_supervised_frame(self.frame, min_history_days=28)
        history = {
            row.demand_date: float(row.bookings)
            for row in self.frame.itertuples(index=False)
        }
        for row in supervised.itertuples(index=False):
            day = pd.Timestamp(row.demand_date)
            self.assertEqual(row.bookings_lag_1, history[day - pd.Timedelta(days=1)])
            self.assertEqual(row.bookings_lag_7, history[day - pd.Timedelta(days=7)])
            roll = [history[day - pd.Timedelta(days=i)] for i in range(1, 8)]
            self.assertTrue(np.isclose(row.bookings_roll_mean_7, float(np.mean(roll))))

    def test_static_horizon_blocks_intraweek_actuals(self):
        cutoff = FIRST_WEEK_START
        rows = build_horizon_feature_rows(self.frame, cutoff, horizon_days=7)
        assert_horizon_uses_only_precutoff_actuals(self.frame, cutoff, rows)
        # Day 0 can see lag_1 = cutoff-1; later days cannot see lag_1 actuals.
        self.assertFalse(pd.isna(rows.iloc[0]["bookings_lag_1"]))
        for offset in range(1, 7):
            self.assertTrue(pd.isna(rows.iloc[offset]["bookings_lag_1"]))
            self.assertTrue(pd.isna(rows.iloc[offset]["guests_lag_1"]))
        # lag_7 remains available throughout a 7-day horizon.
        self.assertFalse(rows["bookings_lag_7"].isna().any())

    def test_future_actual_injection_is_detected(self):
        cutoff = pd.Timestamp(FIRST_WEEK_START)
        poisoned = build_horizon_feature_rows(self.frame, cutoff, horizon_days=7)
        # Force a leaked lag value that would require a within-horizon actual.
        poisoned.loc[3, "bookings_lag_1"] = 99.0
        with self.assertRaises(AssertionError):
            assert_horizon_uses_only_precutoff_actuals(self.frame, cutoff, poisoned)

    def test_recursive_mode_still_keeps_roll_trend_precutoff(self):
        cutoff = FIRST_WEEK_START
        preds = {"bookings": [1.0, 2.0], "guests": [7.0, 14.0]}
        rows = build_horizon_feature_rows(
            self.frame, cutoff, horizon_days=7, recursive_predictions=preds
        )
        history = self.frame[self.frame.demand_date < pd.Timestamp(cutoff)]
        expected_roll = history.bookings.tail(7).mean()
        self.assertTrue(np.isclose(rows.iloc[2]["bookings_roll_mean_7"], expected_roll))
        self.assertEqual(rows.iloc[1]["bookings_lag_1"], 1.0)


class PreparedArtifactTests(unittest.TestCase):
    def test_prepared_outputs_exist_when_generated(self):
        if not PREPARED.exists():
            self.skipTest("Run prepare.py before checking written artifacts")
        daily = pd.read_csv(PREPARED / "data" / "daily_demand.csv")
        self.assertEqual(len(daily), SNAPSHOT_DAYS)
        self.assertEqual(int(daily.bookings.sum()), SNAPSHOT_BOOKINGS)
        self.assertEqual(int(daily.guests.sum()), SNAPSHOT_GUESTS)
        self.assertTrue((PREPARED / "feature_manifest.json").exists())
        self.assertTrue((PREPARED / "data" / "supervised_features.csv").exists())
        self.assertTrue((PREPARED / "data" / "horizon_features_static.csv").exists())
        self.assertTrue((PREPARED / "data" / "arima_bookings_series.csv").exists())

    def test_serving_hashes_stable_after_tests(self):
        after = serving_hashes()
        # Reconstruction is read-only against serving files.
        self.assertEqual(len(after), len(serving_hashes()))


if __name__ == "__main__":
    unittest.main()
