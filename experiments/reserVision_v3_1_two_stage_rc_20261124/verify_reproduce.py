"""Reproduce frozen two-stage rolling/holdout metrics from intermittent_bookings."""
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
SOURCE = ROOT / "experiments/multimodel_dataset_20261123/prepared/data/daily_demand.csv"
sys.path.insert(0, str(ROOT))

from evaluate_forecasts import load_history, metrics  # noqa: E402

spec = importlib.util.spec_from_file_location("intermittent", PREV / "experiment.py")
intermittent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intermittent)


class ReproduceTwoStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = json.loads((HERE / "package/frozen_selection.json").read_text(encoding="utf-8"))
        cls.history = load_history(SOURCE).set_index("demand_date").bookings.astype(float)
        cls.saved = pd.read_csv(HERE / "package/evaluation_evidence/daily_predictions.csv")
        cls.params = {
            (s["protocol"], s["origin"]): s["parameters"]
            for s in cls.freeze["per_origin_parameters"]
            if s["method"] == "two_stage"
        }

    def test_version_and_scope(self):
        self.assertEqual(self.freeze["reservision_version"], "3.1")
        self.assertEqual(self.freeze["selected_experimental_booking_model"], "two_stage")
        self.assertEqual(self.freeze["deployment_scope"], "release-candidate-not-deployed")
        self.assertFalse(self.freeze["production_approved"])
        self.assertTrue(self.freeze["guests"]["preserved"])
        self.assertEqual(self.freeze["guests"]["model"], "robust_median_guests")

    def test_reproduce_rolling_weekly_headline(self):
        g = self.saved[(self.saved.protocol == "rolling_7day") & (self.saved.method == "two_stage")]
        weekly = g.groupby(["origin", "week"])[["actual", "predicted"]].sum()
        m = metrics(weekly.actual, weekly.predicted)
        self.assertAlmostEqual(m["wape"], 32.72234791049451, places=10)
        self.assertAlmostEqual(m["mae"], 2.769713019566857, places=10)
        self.assertAlmostEqual(m["rmse"], 3.909415622750741, places=10)
        self.assertAlmostEqual(m["bias"], 0.9403568384502454, places=10)

    def test_regenerate_matches_saved_predictions(self):
        rows = []
        for (protocol, origin), cfg in sorted(self.params.items()):
            origin_ts = pd.Timestamp(origin)
            horizon = 7 if protocol == "rolling_7day" else 28
            train = self.history[self.history.index < origin_ts]
            test = self.history.loc[origin_ts : origin_ts + pd.Timedelta(days=horizon - 1)]
            pred = intermittent.predict(train, test.index, "two_stage", cfg)
            for day, a, p in zip(test.index, test.to_numpy(float), pred):
                rows.append(dict(protocol=protocol, origin=origin, date=str(day.date()), actual=a, predicted=p))
        regen = pd.DataFrame(rows)
        saved = self.saved[self.saved.method == "two_stage"][
            ["protocol", "origin", "date", "actual", "predicted"]
        ].sort_values(["protocol", "origin", "date"])
        regen = regen.sort_values(["protocol", "origin", "date"])
        pd.testing.assert_frame_equal(
            regen.reset_index(drop=True),
            saved.reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
        )

    def test_holdout_preserved(self):
        g = self.saved[(self.saved.protocol == "original_holdout_28day") & (self.saved.method == "two_stage")]
        weekly = g.groupby(["origin", "week"])[["actual", "predicted"]].sum()
        m = metrics(weekly.actual, weekly.predicted)
        self.assertAlmostEqual(m["wape"], 143.75, places=8)
        self.assertEqual(len(g), 28)

    def test_canonical_unchanged(self):
        integrity = json.loads((HERE / "source_integrity.json").read_text(encoding="utf-8"))
        now = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in integrity["canonical_sha256"]
        }
        self.assertEqual(now, integrity["canonical_sha256"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReproduceTwoStageTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "tests_run": result.testsRun,
        "failures": len(result.failures) + len(result.errors),
        "passed": result.wasSuccessful(),
        "reproduced_weekly_wape": 32.72234791049451,
    }
    (HERE / "verification_reproduce.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sys.exit(not result.wasSuccessful())
