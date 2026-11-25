"""Verify V3.1 RC with floor_round: metrics, consistency, guests impact, horizons."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / "experiments/intermittent_bookings_20261124"
OPS = ROOT / "experiments/v3_1_postprocessing_optimization_20261124"
SOURCE = ROOT / "experiments/multimodel_dataset_20261123/prepared/data/daily_demand.csv"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPS))

from evaluate_forecasts import load_history, metrics
import predict_real as serving
from ops import apply_ops

spec = importlib.util.spec_from_file_location("intermittent", PREV / "experiment.py")
intermittent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intermittent)


def robust_guests(history: pd.Series, dates) -> np.ndarray:
    bundle = {
        "model_type": serving.ROBUST_MEDIAN_GUESTS,
        "baseline_history": [{"ds": str(ts.date()), "y": float(v)} for ts, v in history.items()],
    }
    return np.asarray(
        [r["prediction"] for r in serving.predict_series(bundle, pd.DatetimeIndex(dates))],
        float,
    )


class FloorRoundRCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = json.loads((HERE / "package/frozen_selection.json").read_text(encoding="utf-8"))
        cls.ops_daily = pd.read_csv(HERE / "package/postprocess_evidence/daily_predictions.csv")
        cls.history_b = load_history(SOURCE).set_index("demand_date").bookings.astype(float)
        cls.history_g = load_history(SOURCE).set_index("demand_date").guests.astype(float)
        cls.params = {
            s["origin"]: s["parameters"]
            for s in cls.freeze["per_origin_parameters"]
            if s["method"] == "two_stage" and s["protocol"] == "rolling_7day"
        }

    def test_freeze_includes_floor_round(self):
        self.assertEqual(self.freeze["configuration_complete"]["bookings"], "two_stage")
        self.assertEqual(self.freeze["configuration_complete"]["guests"], "robust_median_guests")
        self.assertEqual(self.freeze["configuration_complete"]["postprocess"], "floor_round")
        self.assertEqual(self.freeze["postprocess"]["selected_mode"], "floor_round")
        self.assertFalse(self.freeze["postprocess"]["party_size_floor"])
        self.assertFalse(self.freeze["production_approved"])
        self.assertAlmostEqual(self.freeze["postprocess"]["nested_past_only_selected_weekly_wape"], 35.0210970464135, places=8)

    def test_reproduce_raw_weekly_wape(self):
        g = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "raw_float")]
        weekly = g.groupby(["origin", "week"])[["actual_bookings", "pred_bookings"]].sum()
        m = metrics(weekly.actual_bookings, weekly.pred_bookings)
        self.assertAlmostEqual(m["wape"], 32.72234791049451, places=10)

    def test_reproduce_floor_round_bookings_weekly(self):
        g = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "floor_round")]
        weekly = g.groupby(["origin", "week"])[["actual_bookings", "pred_bookings"]].sum()
        m = metrics(weekly.actual_bookings, weekly.pred_bookings)
        self.assertAlmostEqual(m["wape"], 29.957805907172997, places=10)
        self.assertAlmostEqual(m["mae"], 2.5357142857142856, places=10)
        # Exact reproduce from evaluate_forecasts.metrics on weekly aggregates.
        self.assertAlmostEqual(m["rmse"], 3.8683699040145725, places=10)
        self.assertAlmostEqual(m["bias"], -0.4642857142857143, places=10)
        # Headline rounded claim (29.96 / 2.536 / 3.868 / -0.464).
        self.assertAlmostEqual(m["wape"], 29.96, places=2)
        self.assertAlmostEqual(m["mae"], 2.536, places=3)
        self.assertAlmostEqual(m["rmse"], 3.868, places=3)
        self.assertAlmostEqual(m["bias"], -0.464, places=3)

    def test_preserve_nested_selection_35_02(self):
        g = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "selected_past_only")]
        weekly = g.groupby(["origin", "week"])[["actual_bookings", "pred_bookings"]].sum()
        m = metrics(weekly.actual_bookings, weekly.pred_bookings)
        self.assertAlmostEqual(m["wape"], 35.021097046413505, places=8)

    def test_floor_round_consistency_and_zero_booking_positive_guest_policy(self):
        g = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "floor_round")]
        for row in g.itertuples(index=False):
            b, gu = int(row.pred_bookings), int(row.pred_guests)
            self.assertGreaterEqual(b, 0)
            self.assertGreaterEqual(gu, 0)
            self.assertGreaterEqual(gu, b)
            if gu > 0:
                self.assertGreaterEqual(b, 1)
        # Regenerate a sample week and confirm apply_ops floor_round matches saved ints.
        origin = "2026-05-04"
        cfg = self.params[origin]
        origin_ts = pd.Timestamp(origin)
        train_b = self.history_b[self.history_b.index < origin_ts]
        train_g = self.history_g[self.history_g.index < origin_ts]
        dates = pd.date_range(origin_ts, periods=7)
        raw_b = intermittent.predict(train_b, dates, "two_stage", cfg)
        raw_g = robust_guests(train_g, dates)
        op_b, op_g = apply_ops(raw_b, raw_g, mode="floor_round")
        saved = g[g.origin == origin].sort_values("date")
        self.assertEqual(list(saved.pred_bookings.astype(int)), op_b)
        self.assertEqual(list(saved.pred_guests.astype(int)), op_g)

    def test_guest_accuracy_not_harmed_vs_raw(self):
        raw = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "raw_float")]
        floor = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "floor_round")]
        var_a = self.ops_daily[(self.ops_daily.model == "two_stage") & (self.ops_daily.ops == "variant_a")]
        raw_m = metrics(
            raw.groupby(["origin", "week"]).actual_guests.sum(),
            raw.groupby(["origin", "week"]).pred_guests.sum(),
        )
        floor_m = metrics(
            floor.groupby(["origin", "week"]).actual_guests.sum(),
            floor.groupby(["origin", "week"]).pred_guests.sum(),
        )
        a_m = metrics(
            var_a.groupby(["origin", "week"]).actual_guests.sum(),
            var_a.groupby(["origin", "week"]).pred_guests.sum(),
        )
        # Floor_round must not materially worsen guests vs raw; Variant A is worse.
        self.assertLessEqual(floor_m["wape"], raw_m["wape"] + 0.05)
        self.assertLess(floor_m["wape"], a_m["wape"])
        self.__class__.guest_impact = dict(raw=raw_m, floor_round=floor_m, variant_a=a_m)

    def test_horizons_7_30_90(self):
        cutoff = self.history_b.index.max() + pd.Timedelta(days=1)
        train_b = self.history_b[self.history_b.index < cutoff]
        train_g = self.history_g[self.history_g.index < cutoff]
        cfg = intermittent.tune(train_b, "two_stage", 7)
        snaps = {}
        for days in (7, 30, 90):
            dates = pd.date_range(cutoff, periods=days)
            raw_b = intermittent.predict(train_b, dates, "two_stage", cfg)
            raw_g = robust_guests(train_g, dates)
            op_b, op_g = apply_ops(raw_b, raw_g, mode="floor_round")
            self.assertEqual(len(op_b), days)
            for b, g in zip(op_b, op_g):
                self.assertIs(type(b), int)
                self.assertIs(type(g), int)
                self.assertGreaterEqual(g, b)
                if g > 0:
                    self.assertGreaterEqual(b, 1)
            snaps[str(days)] = dict(
                bookings_sum=int(sum(op_b)),
                guests_sum=int(sum(op_g)),
                start=str(cutoff.date()),
                end=str(dates[-1].date()),
            )
        self.__class__.horizons = snaps

    def test_canonical_unchanged(self):
        integrity = json.loads((HERE / "source_integrity.json").read_text(encoding="utf-8"))
        now = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in integrity["canonical_sha256"]
        }
        self.assertEqual(now, integrity["canonical_sha256"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FloorRoundRCTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "tests_run": result.testsRun,
        "failures": len(result.failures) + len(result.errors),
        "passed": result.wasSuccessful(),
        "retrospective": {
            "raw_weekly_wape": 32.72234791049451,
            "floor_round_weekly_wape": 29.957805907172997,
            "floor_round_weekly_mae": 2.5357142857142856,
            "floor_round_weekly_rmse": 3.8683699040145725,
            "floor_round_weekly_bias": -0.4642857142857143,
            "nested_selected_weekly_wape": 35.021097046413505,
        },
        "guest_impact": getattr(FloorRoundRCTests, "guest_impact", None),
        "horizons": getattr(FloorRoundRCTests, "horizons", None),
        "prospective_validation": "not_run_required_before_production",
        "production_approved": False,
    }
    # Convert metrics dicts to plain floats
    if payload["guest_impact"]:
        payload["guest_impact"] = {
            k: {mk: float(mv) if mv is not None else None for mk, mv in v.items()}
            for k, v in payload["guest_impact"].items()
        }
    (HERE / "verification_floor_round.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sys.exit(not result.wasSuccessful())
