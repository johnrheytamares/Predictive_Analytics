"""Isolated ASGI verification of the release-candidate artifacts."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
os.environ["PREDICTION_ARTIFACT_DIR"] = str(ARTIFACTS)
sys.path.insert(0, str(ROOT))

import pandas as pd
import predict_real as p
import api


async def request(path, params=None):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    key = os.getenv("PREDICTION_API_KEY", "").strip()
    headers = [(b"x-api-key", key.encode())] if key else []
    await api.app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": urlencode(params or {}).encode(),
            "headers": headers,
            "client": ("local-rc", 1),
            "server": ("localhost", 80),
            "root_path": "",
        },
        receive,
        send,
    )
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = json.loads(b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body"))
    return start["status"], body, dict(start.get("headers", []))


def get(path, params=None):
    return asyncio.run(request(path, params))


def query(start, end):
    return get("/forecast", {"start_date": start, "end_date": end})


class ReleaseCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshots = {}
        cls.integrity = json.loads((HERE / "source_integrity.json").read_text(encoding="utf-8"))
        cls.canonical = cls.integrity["canonical_sha256_before"]

    @classmethod
    def tearDownClass(cls):
        now = {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
            for n in cls.canonical
        }
        assert now == cls.canonical

    def test_health_dispatch_policy_and_scope(self):
        status, health, _ = get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["model_ready"])
        meta = health["metadata"]
        self.assertEqual(meta["deployment_scope"], "release-candidate-not-deployed")
        self.assertFalse(meta.get("production_approved"))
        self.assertEqual(meta["targets"]["bookings"]["selected_model"], "weekday_baseline")
        self.assertEqual(meta["targets"]["guests"]["selected_model"], p.ROBUST_MEDIAN_GUESTS)
        self.assertEqual(meta["targets"]["guests"].get("interval_status"), "Not evaluated")
        self.assertFalse(meta["targets"]["guests"].get("interval_coverage_evaluated"))
        caps = health["forecast_capabilities"]
        self.assertEqual(caps["profile"], "cutoff_anchored_four_week_weekday_median")
        self.assertEqual(caps["maximum_days_per_cutoff"], p.MAX_FORECAST_DAYS)
        self.assertEqual(len(caps["supported_dates"]), p.MAX_FORECAST_DAYS)
        self.assertFalse(caps["recursive_predictions"])
        self.assertIsNone(caps["fallback"])
        # Embedded metadata policy must match live capabilities on max days/profile.
        policy = meta["forecast_policy"]
        self.assertEqual(policy["maximum_days_per_cutoff"], p.MAX_FORECAST_DAYS)
        self.assertEqual(policy["profile"], caps["profile"])
        self.assertEqual(policy.get("interval_status"), "Not evaluated")
        b, g = p.get_model_bundles(use_cache=False)
        self.assertEqual((b["model_type"], g["model_type"]), ("weekday_baseline", p.ROBUST_MEDIAN_GUESTS))
        self.__class__.snapshots["health"] = {
            "deployment_scope": meta["deployment_scope"],
            "bookings": meta["targets"]["bookings"]["selected_model"],
            "guests": meta["targets"]["guests"]["selected_model"],
            "capabilities": caps,
            "interval_status": meta["targets"]["guests"].get("interval_status"),
        }

    def test_ranges_variant_a_and_demographics(self):
        for end, days in (("2026-11-28", 7), ("2026-12-21", 30), ("2027-02-19", 90)):
            status, body, _ = query("2026-11-22", end)
            self.assertEqual(status, 200)
            self.assertEqual(len(body["forecast"]), days)
            self.assertEqual(body["model"]["targets"]["bookings"]["selected_model"], "weekday_baseline")
            self.assertEqual(body["model"]["targets"]["guests"]["selected_model"], p.ROBUST_MEDIAN_GUESTS)
            for row in body["forecast"]:
                b, g = row["predicted_bookings"], row["predicted_total_guests"]
                self.assertIs(type(b), int)
                self.assertIs(type(g), int)
                self.assertGreaterEqual(g, 7 * b)
                if g > 0:
                    self.assertGreaterEqual(b, 1)
                self.assertEqual(sum(row["estimated_guest_mix"].values()), g)
                self.assertLessEqual(row["guests_lower"], g)
                self.assertGreaterEqual(row["guests_upper"], g)
            self.__class__.snapshots[f"range_{days}"] = {
                "days": days,
                "bookings": sum(r["predicted_bookings"] for r in body["forecast"]),
                "guests": sum(r["predicted_total_guests"] for r in body["forecast"]),
            }

    def test_no_future_leak_and_profile_reuse(self):
        _, g = p.get_model_bundles(use_cache=False)
        clean = p.predict_series(g, pd.date_range("2026-11-22", periods=8))
        augmented = dict(g, baseline_history=g["baseline_history"] + [{"ds": "2026-11-22", "y": 9999}])
        leaked = p.predict_series(augmented, pd.date_range("2026-11-22", periods=8))
        self.assertEqual(clean, leaked)
        self.assertEqual(clean[0]["prediction"], clean[7]["prediction"])

    def test_unsupported_insufficient_weekday_history(self):
        _, g = p.get_model_bundles(use_cache=False)
        kept = [r for r in g["baseline_history"] if pd.Timestamp(r["ds"]).dayofweek != 0]
        mondays = [r for r in g["baseline_history"] if pd.Timestamp(r["ds"]).dayofweek == 0][-3:]
        damaged = dict(g, baseline_history=kept + mondays)
        with self.assertRaises(p.ForecastError) as ctx:
            p.predict_series(damaged, pd.date_range("2026-11-23", periods=1))
        self.assertEqual(ctx.exception.code, "ROBUST_RANGE_UNSUPPORTED")
        detail = ctx.exception.extra["unsupported_dates"][0]
        self.assertEqual(detail["available_matching_weekdays"], 3)
        self.assertFalse(ctx.exception.extra["policy"]["recursive_predictions"])
        self.__class__.snapshots["unsupported"] = {
            "code": ctx.exception.code,
            "available": detail["available_matching_weekdays"],
        }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseCandidateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        payload = {
            "step": "2F",
            "tests_passed": result.testsRun,
            "failures": len(result.failures) + len(result.errors),
            "canonical_artifacts_unchanged": True,
            "responses": ReleaseCandidateTests.snapshots,
        }
        (HERE / "verification_fastapi.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    sys.exit(not result.wasSuccessful())
