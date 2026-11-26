"""Regression tests for isolated v3.2 sumratio guests RC."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V31 = ROOT / "experiments" / "reserVision_v3_1_two_stage_rc_20261124"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class V32SumratioRCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v32 = load_module(HERE / "shadow_predict.py", "v32_rc_test")
        cls.v31 = load_module(V31 / "shadow_predict.py", "v31_rc_test")
        cls.v32.clear_cache()
        cls.v31.clear_cache()
        cls.verification = json.loads((HERE / "verification_results.json").read_text(encoding="utf-8"))
        cls.meta = json.loads((HERE / "artifacts" / "forecast_models_metadata.json").read_text(encoding="utf-8"))

    def test_metadata_isolation(self):
        self.assertEqual(self.meta["model_version"], "3.2")
        self.assertIs(self.meta["production_approved"], False)
        self.assertIs(self.meta["shadow_serving_only"], True)
        self.assertEqual(self.meta["targets"]["guests"]["selected_model"], "guests_per_booking_sumratio")
        self.assertEqual(self.meta["targets"]["bookings"]["selected_model"], "two_stage")
        self.assertFalse(self.meta["forecast_policy"]["party_size_floor"])
        self.assertFalse(self.meta["forecast_policy"]["holiday_multipliers"])

    def test_weekly_metrics_reproduced(self):
        self.assertAlmostEqual(self.verification["weekly_guest_mae"], 29.18, places=2)
        self.assertAlmostEqual(self.verification["weekly_guest_rmse"], 40.94, places=2)
        self.assertAlmostEqual(self.verification["weekly_booking_wape"], 29.96, places=2)

    def test_sample_jan_feb_2027(self):
        payload = self.v32.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
        bookings = [r["predicted_bookings"] for r in payload["forecast"]]
        guests = [r["predicted_total_guests"] for r in payload["forecast"]]
        self.assertEqual(sum(bookings), 12)
        self.assertEqual(sum(guests), 108)
        v31 = self.v31.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
        self.assertEqual(bookings, [r["predicted_bookings"] for r in v31["forecast"]])
        self.assertEqual(sum(r["predicted_total_guests"] for r in v31["forecast"]), 18)

    def test_zero_demand_and_demographic_consistency(self):
        payload = self.v32.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
        from predict_real import GUEST_MIX_CATEGORIES, estimate_guest_mix

        agg = {k: 0 for k in GUEST_MIX_CATEGORIES}
        total_guests = 0
        for row in payload["forecast"]:
            b, g, mix = row["predicted_bookings"], row["predicted_total_guests"], row["estimated_guest_mix"]
            if b == 0:
                self.assertEqual(g, 0)
                self.assertEqual(sum(mix.values()), 0)
            else:
                self.assertGreaterEqual(g, b)
            self.assertEqual(sum(mix.values()), g)
            total_guests += g
            for k, v in mix.items():
                agg[k] += v
        self.assertEqual(agg, estimate_guest_mix(total_guests, payload["model_metadata"]))

    def test_sumratio_intervals_not_robust_median(self):
        payload = self.v32.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
        v31 = self.v31.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
        # At least one day with bookings should differ in guest interval width or point
        differed = False
        for a, b in zip(payload["forecast"], v31["forecast"]):
            if a["predicted_bookings"] > 0:
                if (
                    a["predicted_total_guests"] != b["predicted_total_guests"]
                    or a["guests_lower"] != b["guests_lower"]
                    or a["guests_upper"] != b["guests_upper"]
                ):
                    differed = True
                    self.assertLessEqual(a["guests_lower"], a["predicted_total_guests"])
                    self.assertGreaterEqual(a["guests_upper"], a["predicted_total_guests"])
        self.assertTrue(differed)

    def test_horizons_7_30_90_supported(self):
        for days, end in [(7, "2027-01-07"), (30, "2027-01-30"), (90, "2027-03-31")]:
            payload = self.v32.run_forecast("2027-01-01", end, use_cache=False)
            self.assertEqual(payload["forecast_range"]["days"], days)
            self.assertEqual(len(payload["forecast"]), days)

    def test_integrity_and_rollback_pointer(self):
        snap = self.v32.validate_integrity()
        self.assertEqual(snap["rollback_release_candidate_id"], "reservision_v3_1_two_stage_rc_20261124")
        self.assertEqual(self.meta["rollback_release"]["release_candidate_id"], "reservision_v3_1_two_stage_rc_20261124")


if __name__ == "__main__":
    unittest.main()
