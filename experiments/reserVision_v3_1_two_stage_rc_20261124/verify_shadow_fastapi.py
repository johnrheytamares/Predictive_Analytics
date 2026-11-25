"""ASGI integration + regression tests for isolated V3.1 shadow FastAPI."""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Validate existing artifacts only. Never regenerate a frozen release in a test.
import shadow_api  # noqa: E402
import shadow_predict as sp  # noqa: E402
import predict_real as production_predict  # noqa: E402

# Production response key contract (compatibility).
PRODUCTION_FORECAST_KEYS = {
    "success",
    "model_version",
    "generated_at",
    "model",
    "model_metadata",
    "forecast_range",
    "guest_mix",
    "aggregate_interval_available",
    "forecast",
}
PRODUCTION_ROW_KEYS = {
    "date",
    "predicted_bookings",
    "bookings_lower",
    "bookings_upper",
    "predicted_total_guests",
    "guests_lower",
    "guests_upper",
    "estimated_guest_mix",
    "guest_mix_ready",
}


async def request(path, params=None):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    key = os.getenv("PREDICTION_API_KEY", "").strip()
    headers = [(b"x-api-key", key.encode())] if key else []
    await shadow_api.app(
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
            "client": ("local-v31-shadow", 1),
            "server": ("localhost", 8002),
            "root_path": "",
        },
        receive,
        send,
    )
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = json.loads(
        b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    )
    return start["status"], body, dict(start.get("headers", []))


def get(path, params=None):
    return asyncio.run(request(path, params))


def query(start, end, **extra):
    params = {"start_date": start, "end_date": end}
    params.update(extra)
    return get("/forecast", params)


class ShadowFastAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshots = {}
        cls.integrity = json.loads((HERE / "source_integrity.json").read_text(encoding="utf-8"))
        cls.canonical = cls.integrity["canonical_sha256"]
        sp.clear_cache()

    @classmethod
    def tearDownClass(cls):
        now = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in cls.canonical}
        assert now == cls.canonical, "Canonical production artifacts were modified"

    def test_health_isolated_configuration(self):
        status, health, _ = get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["model_ready"])
        self.assertTrue(health["shadow_serving_only"])
        self.assertFalse(health["production_approved"])
        self.assertEqual(health["service"], "predictive-analytics-v3.1-shadow")
        meta = health["metadata"]
        self.assertEqual(meta["deployment_scope"], "release-candidate-not-deployed")
        self.assertEqual(meta["model_version"], "3.1")
        self.assertFalse(meta.get("production_approved"))
        self.assertEqual(meta["targets"]["bookings"]["selected_model"], "two_stage")
        self.assertEqual(meta["targets"]["guests"]["selected_model"], "robust_median_guests")
        self.assertEqual(meta["postprocess"]["selected_mode"], "floor_round")
        self.assertEqual(health["selected_configuration"]["postprocess"], "floor_round")
        # Separate booking and guest weekly metrics.
        bw = health["weekly_performance"]["bookings"]
        gw = health["weekly_performance"]["guests"]
        self.assertAlmostEqual(bw["raw"]["wape"], 32.72234791049451, places=8)
        self.assertAlmostEqual(bw["operational_floor_round"]["wape"], 29.957805907173, places=8)
        self.assertAlmostEqual(bw["operational_floor_round"]["mae"], 2.5357142857142856, places=8)
        self.assertAlmostEqual(bw["operational_floor_round"]["rmse"], 3.868369904014573, places=8)
        self.assertAlmostEqual(bw["operational_floor_round"]["bias"], -0.4642857142857143, places=8)
        self.assertIn("wape", gw["raw"])
        self.assertIn("wape", gw["operational_floor_round"])
        self.assertEqual(health["evaluation_scope"], "retrospective_rolling_28_origins")
        self.assertEqual(
            health["prospective_validation"], "not_run_required_before_production"
        )
        caps = health["forecast_capabilities"]
        self.assertEqual(caps["postprocess"], "floor_round")
        self.assertEqual(caps["bookings_method"], "two_stage")
        self.assertFalse(caps["party_size_floor"])
        self.assertEqual(caps["maximum_days_per_cutoff"], 90)
        self.assertGreaterEqual(len(caps["supported_dates"]), 90)
        self.__class__.snapshots["health"] = {
            "service": health["service"],
            "bookings": meta["targets"]["bookings"]["selected_model"],
            "guests": meta["targets"]["guests"]["selected_model"],
            "postprocess": meta["postprocess"]["selected_mode"],
            "weekly_performance": health["weekly_performance"],
            "evaluation_scope": health["evaluation_scope"],
            "prospective_validation": health["prospective_validation"],
        }

    def test_forecast_compatibility_and_horizons(self):
        for end, days in (("2026-11-28", 7), ("2026-12-21", 30), ("2027-02-19", 90)):
            status, body, _ = query("2026-11-22", end)
            self.assertEqual(status, 200)
            self.assertTrue(PRODUCTION_FORECAST_KEYS.issubset(body.keys()))
            self.assertEqual(body["model_version"], "3.1")
            self.assertEqual(len(body["forecast"]), days)
            self.assertEqual(body["model"]["targets"]["bookings"]["selected_model"], "two_stage")
            self.assertEqual(
                body["model"]["targets"]["guests"]["selected_model"], "robust_median_guests"
            )
            self.assertEqual(body["model"]["postprocess"]["selected_mode"], "floor_round")
            self.assertTrue(body["model"]["shadow_serving_only"])
            self.assertFalse(body["model"]["production_approved"])
            # Separate weekly metrics exposed on forecast responses.
            self.assertIn("bookings_weekly_metrics", body["model"])
            self.assertIn("guests_weekly_metrics", body["model"])
            self.assertAlmostEqual(
                body["model"]["bookings_weekly_metrics"]["raw"]["wape"],
                32.72234791049451,
                places=8,
            )
            self.assertAlmostEqual(
                body["model"]["bookings_weekly_metrics"]["operational_floor_round"]["wape"],
                29.957805907173,
                places=8,
            )
            for row in body["forecast"]:
                self.assertTrue(PRODUCTION_ROW_KEYS.issubset(row.keys()))
                b, g = row["predicted_bookings"], row["predicted_total_guests"]
                self.assertIs(type(b), int)
                self.assertIs(type(g), int)
                self.assertGreaterEqual(b, 0)
                self.assertGreaterEqual(g, 0)
                self.assertGreaterEqual(g, b)
                if g > 0:
                    self.assertGreaterEqual(b, 1)
                if row["estimated_guest_mix"] is not None:
                    self.assertEqual(sum(row["estimated_guest_mix"].values()), g)
                self.assertLessEqual(row["guests_lower"], g)
                self.assertGreaterEqual(row["guests_upper"], g)
            self.__class__.snapshots[f"range_{days}"] = {
                "days": days,
                "bookings": sum(r["predicted_bookings"] for r in body["forecast"]),
                "guests": sum(r["predicted_total_guests"] for r in body["forecast"]),
                "compatible_keys": True,
            }

    def test_variant_a_guest_reduction_explained(self):
        status, body, _ = query("2026-11-22", "2026-11-28", compare_variant_a="true")
        self.assertEqual(status, 200)
        cmp_ = body["postprocess_comparison"]
        floor_g = cmp_["floor_round"]["guests_sum"]
        var_g = cmp_["variant_a"]["guests_sum"]
        self.assertLess(floor_g, var_g)
        self.assertEqual(cmp_["guest_reduction_vs_variant_a"], var_g - floor_g)
        self.assertIn("party floor", cmp_["explanation"].lower())
        # Match prior horizon evidence: Variant A guests inflated by party floor.
        self.assertEqual(var_g, 14)
        self.assertEqual(floor_g, 3)
        self.__class__.snapshots["variant_a_comparison_7d"] = cmp_

        # Also check 30/90 totals for the report.
        for end, days, expected_floor, expected_var in (
            ("2026-12-21", 30, 14, 63),
            ("2027-02-19", 90, 38, 175),
        ):
            _, body_h, _ = query("2026-11-22", end, compare_variant_a="true")
            c = body_h["postprocess_comparison"]
            self.assertEqual(c["floor_round"]["guests_sum"], expected_floor)
            self.assertEqual(c["variant_a"]["guests_sum"], expected_var)
            self.__class__.snapshots[f"variant_a_comparison_{days}d"] = c

    def test_production_api_untouched_and_regression_contract(self):
        # Production module still present and still enforces Variant A-like party floor path.
        self.assertTrue(hasattr(production_predict, "run_forecast"))
        self.assertTrue(hasattr(production_predict, "generate_forecast"))
        src = Path(production_predict.__file__).read_text(encoding="utf-8")
        self.assertIn("typical_party_size", src)
        # Shadow app is a distinct service identity.
        self.assertNotEqual(shadow_api.app.title, "ReserVision Predictive Analytics")
        self.assertIn("Shadow", shadow_api.app.title)
        # Canonical root artifacts unchanged (also asserted in tearDownClass).
        now = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in self.canonical}
        self.assertEqual(now, self.canonical)
        self.__class__.snapshots["production_untouched"] = True

    def test_evidence_preserved(self):
        for rel in (
            "package/evaluation_evidence/results.json",
            "package/evaluation_evidence/daily_predictions.csv",
            "package/postprocess_evidence/comparison.csv",
            "package/frozen_selection.json",
            "verification_floor_round.json",
        ):
            path = HERE / rel
            self.assertTrue(path.exists(), rel)
            self.assertGreater(path.stat().st_size, 0)
        freeze = json.loads((HERE / "package/frozen_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(freeze["configuration_complete"]["postprocess"], "floor_round")
        self.assertAlmostEqual(
            freeze["postprocess"]["nested_past_only_selected_weekly_wape"],
            35.0210970464135,
            places=8,
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ShadowFastAPITests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "suite": "verify_shadow_fastapi",
        "tests_run": result.testsRun,
        "failures": len(result.failures) + len(result.errors),
        "passed": result.wasSuccessful(),
        "canonical_artifacts_unchanged": True,
        "production_service_replaced": False,
        "responses": ShadowFastAPITests.snapshots,
        "api_compatibility": {
            "endpoints": ["/health", "/forecast"],
            "forecast_top_level_keys_required": sorted(PRODUCTION_FORECAST_KEYS),
            "forecast_row_keys_required": sorted(PRODUCTION_ROW_KEYS),
            "model_version": "3.1",
            "notes": (
                "Response shape matches production contract keys. "
                "Selected models and postprocess differ (two_stage + floor_round). "
                "Additional weekly metric blocks and optional Variant A comparison are additive."
            ),
        },
        "guest_total_reduction_vs_variant_a": {
            "cause": "ops_policy_not_guest_model_change",
            "detail": (
                "Variant A multiplies bookings by typical party size (~7) as a guest floor; "
                "floor_round does not. Robust Median guest predictions are unchanged."
            ),
        },
        "readiness": {
            "isolated_shadow_serving": True,
            "production_go": False,
            "prospective_validation": "not_run",
        },
    }
    output = HERE / "integration_validation" / "legacy_regression.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    sys.exit(not result.wasSuccessful())
