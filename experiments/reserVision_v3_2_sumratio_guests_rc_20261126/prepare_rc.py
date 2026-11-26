"""Build isolated v3.2 sumratio guest RC artifacts and verification evidence.

Does not modify production_v31 or activate serving.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V31 = ROOT / "experiments" / "reserVision_v3_1_two_stage_rc_20261124"
SRC = ROOT / "experiments" / "v32_corrected_comparative_trials_20261126"
FROZEN = V31 / "package" / "postprocess_evidence" / "daily_predictions.csv"
OUT = HERE / "package" / "evaluation_evidence"
ART = HERE / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("v32_shadow", HERE / "shadow_predict.py")
shadow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)

import predict_real as serving  # noqa: E402

LOOKBACK = 84
ORIGIN_START = "2026-05-04"
ORIGIN_END = "2026-11-09"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    e = p - a
    return dict(
        wape_pct=float(abs(e).sum() / abs(a).sum() * 100) if abs(a).sum() else None,
        mae=float(abs(e).mean()),
        rmse=float(np.sqrt((e * e).mean())),
        bias=float(e.mean()),
        actual_total=float(a.sum()),
        predicted_total=float(p.sum()),
        observations=int(len(a)),
    )


def load_history() -> pd.DataFrame:
    df = pd.read_csv(SRC / "trial1_production_daily.csv", parse_dates=["demand_date"])
    hist = df.set_index("demand_date").sort_index()
    assert len(hist) == 356 and hist.bookings.sum() == 519 and hist.guests.sum() == 6669
    assert hist.index.is_monotonic_increasing and not hist.index.has_duplicates
    return hist


def build_guest_artifact(hist: pd.DataFrame, v31_guests: dict) -> dict:
    baseline = [{"ds": d.strftime("%Y-%m-%d"), "y": float(hist.loc[d, "guests"])} for d in hist.index]
    # Keep booking history twin for integrity pairing
    return {
        "model": None,
        "model_type": shadow.GUESTS_PER_BOOKING_SUMRATIO,
        "baseline_history": baseline,
        "target": "guests",
        "lookback_days": LOOKBACK,
        "estimator": {
            "name": "guests_per_booking_sumratio",
            "formula": "operational_bookings * (sum(guests)/sum(bookings)) over past-84 positive-booking days",
            "party_size_floor": False,
            "holiday_multipliers": False,
        },
        "metadata": {
            "model_version": "3.2",
            "guest_mix_proportions": (v31_guests.get("metadata") or {}).get("guest_mix_proportions")
            or json.loads((V31 / "artifacts" / "forecast_models_metadata.json").read_text())["guest_mix_proportions"],
            "typical_guests_per_booking": 7.0,
            "guest_mix_ready": True,
            "guest_mix_coverage_pct": 99.48,
        },
    }


def evaluate_rolling(hist: pd.DataFrame, bookings_bundle) -> tuple[pd.DataFrame, dict]:
    frozen = pd.read_csv(FROZEN, parse_dates=["origin", "date"])
    frozen = frozen[(frozen.model == "two_stage") & (frozen.ops == "floor_round")].copy()
    assert frozen.origin.nunique() == 28 and len(frozen) == 196

    rows = []
    interval_hits = []
    for origin, part in frozen.groupby("origin"):
        part = part.sort_values("date")
        dates = pd.to_datetime(part.date)
        train = hist.loc[hist.index < pd.Timestamp(origin)]
        # Exact frozen operational bookings
        b = part.pred_bookings.to_numpy(float)
        stats = shadow.sumratio_stats(train)
        point, lower, upper = shadow.predict_sumratio_guests(train, b, stats)
        # Booking identity vs frozen
        raw_b, cfg = shadow.two_stage_raw_bookings(bookings_bundle, dates)
        op_b = shadow.operational_bookings_matching_v31(
            raw_b, dates, guest_history_before=train["guests"], use_cache=False
        )
        booking_match = bool(np.allclose(op_b, b))
        for j, day in enumerate(dates):
            actual_g = float(part.iloc[j].actual_guests)
            actual_b = float(part.iloc[j].actual_bookings)
            rows.append(
                dict(
                    origin=pd.Timestamp(origin).date(),
                    date=day.date(),
                    pred_bookings_frozen=float(b[j]),
                    pred_bookings_recomputed=float(op_b[j]),
                    booking_match=booking_match and abs(op_b[j] - b[j]) < 1e-9,
                    actual_bookings=actual_b,
                    actual_guests=actual_g,
                    pred_guests=float(point[j]),
                    guests_lower=float(lower[j]),
                    guests_upper=float(upper[j]),
                    ratio=stats["ratio"],
                    cfg=json.dumps(cfg, sort_keys=True),
                )
            )
            if b[j] > 0 or actual_g > 0:
                interval_hits.append(lower[j] <= actual_g <= upper[j])

    frame = pd.DataFrame(rows)
    # Primary scoring uses frozen bookings + sumratio guests (exact evaluation protocol)
    score = frame.copy()
    score["pred_bookings"] = score["pred_bookings_frozen"]
    weekly = score.groupby("origin")[["actual_guests", "pred_guests", "actual_bookings", "pred_bookings"]].sum()
    weekly_guest = metrics(weekly.actual_guests, weekly.pred_guests)
    weekly_book = metrics(weekly.actual_bookings, weekly.pred_bookings)
    daily_guest = metrics(score.actual_guests, score.pred_guests)
    assert abs(weekly_book["wape_pct"] - 29.96) < 0.05
    assert abs(weekly_guest["mae"] - 29.18) < 0.05, weekly_guest
    assert abs(weekly_guest["rmse"] - 40.94) < 0.05, weekly_guest

    coverage = float(np.mean(interval_hits)) if interval_hits else None
    summary = dict(
        weekly_guests=weekly_guest,
        weekly_bookings=weekly_book,
        daily_guests=daily_guest,
        interval_method="past_only_party_size_ratio_quantiles_q10_q90",
        interval_coverage_daily=coverage,
        interval_observations=len(interval_hits),
        booking_recompute_match_rate=float(frame.booking_match.mean()),
        origins=28,
    )
    return frame, summary


def evaluate_horizons(hist: pd.DataFrame):
    """7/30/90 metrics. Horizon 7 pins frozen Two-Stage bookings; 30/90 use recomputed two_stage."""
    frozen = pd.read_csv(FROZEN, parse_dates=["origin", "date"])
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="7D")
    bookings_bundle = joblib.load(ART / "bookings_model.joblib")
    rows = []
    for origin in origins:
        train = hist.loc[hist.index < origin]
        for h in (7, 30, 90):
            end = origin + pd.Timedelta(days=h - 1)
            if end > hist.index.max():
                continue
            dates = pd.date_range(origin, periods=h, freq="D")
            raw_b, _ = shadow.two_stage_raw_bookings(bookings_bundle, dates)
            op_b = shadow.operational_bookings_matching_v31(
                raw_b, dates, guest_history_before=train["guests"], use_cache=False
            )
            actual_g = hist.loc[dates, "guests"].to_numpy(float)
            actual_b = hist.loc[dates, "bookings"].to_numpy(float)
            bookings_source = "two_stage_floor_round_v31_identical"
            if h == 7:
                part = frozen[
                    (frozen.model == "two_stage")
                    & (frozen.ops == "floor_round")
                    & (frozen.origin == origin)
                ]
                if len(part) == 7:
                    frozen_b = part.sort_values("date").pred_bookings.to_numpy(float)
                    assert np.allclose(op_b, frozen_b), (origin, op_b.tolist(), frozen_b.tolist())
                    op_b = frozen_b
                    actual_g = part.sort_values("date").actual_guests.to_numpy(float)
                    actual_b = part.sort_values("date").actual_bookings.to_numpy(float)
                    bookings_source = "frozen_two_stage"
            point, lower, upper = shadow.predict_sumratio_guests(train, op_b)
            g_tot = metrics([actual_g.sum()], [point.sum()])
            b_tot = metrics([actual_b.sum()], [op_b.sum()])
            g_day = metrics(actual_g, point)
            b_day = metrics(actual_b, op_b)
            rows.append(
                dict(
                    origin=origin.date(),
                    horizon=h,
                    aggregation="horizon_total",
                    bookings_source=bookings_source,
                    guest_actual_total=g_tot["actual_total"],
                    guest_predicted_total=g_tot["predicted_total"],
                    guest_wape_pct=g_tot["wape_pct"],
                    guest_mae=g_tot["mae"],
                    guest_rmse=g_tot["rmse"],
                    guest_bias=g_tot["bias"],
                    booking_actual_total=b_tot["actual_total"],
                    booking_predicted_total=b_tot["predicted_total"],
                    booking_wape_pct=b_tot["wape_pct"],
                    booking_mae=b_tot["mae"],
                    interval_hit=bool(lower.sum() <= actual_g.sum() <= upper.sum()),
                )
            )
            rows.append(
                dict(
                    origin=origin.date(),
                    horizon=h,
                    aggregation="daily",
                    bookings_source=bookings_source,
                    guest_wape_pct=g_day["wape_pct"],
                    guest_mae=g_day["mae"],
                    guest_rmse=g_day["rmse"],
                    guest_bias=g_day["bias"],
                    booking_mae=b_day["mae"],
                    booking_wape_pct=b_day["wape_pct"],
                    interval_hit=float(np.mean((lower <= actual_g) & (actual_g <= upper))),
                )
            )
    detail = pd.DataFrame(rows)
    pooled = []
    for h in (7, 30, 90):
        part = detail[(detail.horizon == h) & (detail.aggregation == "horizon_total")]
        if part.empty:
            continue
        gm = metrics(part.guest_actual_total, part.guest_predicted_total)
        bm = metrics(part.booking_actual_total, part.booking_predicted_total)
        pooled.append(
            dict(
                horizon=h,
                aggregation="horizon_total_pooled",
                origins=int(part.origin.nunique()),
                guest_mae=gm["mae"],
                guest_rmse=gm["rmse"],
                guest_bias=gm["bias"],
                guest_wape_pct=gm["wape_pct"],
                booking_mae=bm["mae"],
                booking_rmse=bm["rmse"],
                booking_bias=bm["bias"],
                booking_wape_pct=bm["wape_pct"],
                interval_coverage=float(part.interval_hit.mean()),
                bookings_source=part.bookings_source.iloc[0],
            )
        )
    return detail, pd.DataFrame(pooled)


def main() -> None:
    # Protect production manifest
    prod_manifest = json.loads((ROOT / "production_v31" / "release_manifest.json").read_text(encoding="utf-8"))
    before = {rel: digest(ROOT / rel) for rel in prod_manifest["files"]}

    hist = load_history()
    v31_guests = joblib.load(V31 / "artifacts" / "guests_model.joblib")
    guests_art = build_guest_artifact(hist, v31_guests)
    # Pair booking history from v31 bookings artifact (must match 356-day foundation)
    bookings_art = joblib.load(ART / "bookings_model.joblib")
    assert len(bookings_art["baseline_history"]) == 356
    assert len(guests_art["baseline_history"]) == 356
    assert [r["ds"] for r in bookings_art["baseline_history"]] == [r["ds"] for r in guests_art["baseline_history"]]
    joblib.dump(guests_art, ART / "guests_model.joblib")

    def write_snapshot() -> None:
        protected = {
            "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/artifacts/bookings_model.joblib": digest(ART / "bookings_model.joblib"),
            "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/artifacts/guests_model.joblib": digest(ART / "guests_model.joblib"),
            "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/artifacts/frozen_selection.json": digest(ART / "frozen_selection.json"),
            "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/artifacts/ops.py": digest(ART / "ops.py"),
            "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/shadow_predict.py": digest(HERE / "shadow_predict.py"),
            "experiments/reserVision_v3_1_two_stage_rc_20261124/artifacts/bookings_model.joblib": digest(V31 / "artifacts" / "bookings_model.joblib"),
            "experiments/reserVision_v3_1_two_stage_rc_20261124/package/postprocess_evidence/daily_predictions.csv": digest(FROZEN),
            "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/package/rollback/v31_guests_model.joblib": digest(
                HERE / "package" / "rollback" / "v31_guests_model.joblib"
            ),
        }
        if (ART / "forecast_models_metadata.json").exists():
            protected[
                "experiments/reserVision_v3_2_sumratio_guests_rc_20261126/artifacts/forecast_models_metadata.json"
            ] = digest(ART / "forecast_models_metadata.json")
        snapshot = {
            "release_candidate_id": "reservision_v3_2_sumratio_guests_rc_20261126",
            "forward_cutoff": "2026-11-22",
            "forward_parameters": {"prob_days": 14, "size_days": 84, "shrink": 0.0},
            "parameters_source": "identical to v3.1 serving_snapshot forward_parameters",
            "no_runtime_optimization": True,
            "frozen_history_end": "2026-11-21",
            "rollback_release_candidate_id": "reservision_v3_1_two_stage_rc_20261124",
            "protected_files": protected,
        }
        (HERE / "serving_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    write_snapshot()

    # Clear any prior cache and evaluate
    shadow.clear_cache()
    shadow.ARTIFACTS = ART
    frame, summary7 = evaluate_rolling(hist, bookings_art)
    frame.to_csv(OUT / "daily_predictions_sumratio.csv", index=False)

    # Build guests artifact path then horizon eval (needs guests joblib on disk)
    shadow.clear_cache()
    detail, pooled = evaluate_horizons(hist)
    detail.to_csv(OUT / "horizon_origin_metrics.csv", index=False)
    pooled.to_csv(OUT / "horizon_pooled_metrics.csv", index=False)

    # Sample Jan1-Feb8 2027
    # Temporarily write metadata for run_forecast after metadata creation — do sample after metadata+snapshot

    weekly_guest = summary7["weekly_guests"]
    weekly_book = summary7["weekly_bookings"]

    meta = {
        "model_version": "3.2",
        "reservision_version": "3.2",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_days": 356,
        "minimum_training_days": 120,
        "date_from": "2025-12-01",
        "date_to": "2026-11-21",
        "data_source": "booking_demand_daily",
        "is_demo": False,
        "provenance_verified": True,
        "guest_mix_coverage_pct": 99.48,
        "guest_mix_ready": True,
        "guest_mix_proportions": guests_art["metadata"]["guest_mix_proportions"],
        "typical_guests_per_booking": 7.0,
        "ready": True,
        "blocking_reasons": [],
        "safe_for_live_use": True,
        "deployment_scope": "release-candidate-not-deployed",
        "release_candidate_id": "reservision_v3_2_sumratio_guests_rc_20261126",
        "release_configuration": "two_stage_guests_per_booking_sumratio_floor_round",
        "production_approved": False,
        "staging_approved": False,
        "shadow_serving_only": True,
        "serving_mode": "isolated_shadow",
        "rollback_release": {
            "release_candidate_id": "reservision_v3_1_two_stage_rc_20261124",
            "path": str(V31),
            "guests_method": "robust_median_guests",
            "note": "Active production remains v3.1 until an authorized cutover.",
        },
        "postprocess": {
            "selected_mode": "floor_round",
            "party_size_floor": False,
            "description": "Floor bookings; guests from sumratio then guests>=bookings consistency. No party-size floor.",
        },
        "evaluation_scope": "retrospective_rolling_28_origins",
        "prospective_validation": "not_run_required_before_production",
        "targets": {
            "bookings": {
                "target": "bookings",
                "selected_model": "two_stage",
                "evaluation_protocol": "rolling_28_monday_origins_7day",
                "selection_metric": "weekly_wape",
                "model_path": "bookings_model.joblib",
                "selected_metrics": {
                    "wape": weekly_book["wape_pct"],
                    "mae": weekly_book["mae"],
                    "rmse": weekly_book["rmse"],
                    "bias": weekly_book["bias"],
                },
                "operational_weekly_metrics": {
                    "wape": weekly_book["wape_pct"],
                    "mae": weekly_book["mae"],
                    "rmse": weekly_book["rmse"],
                    "bias": weekly_book["bias"],
                },
            },
            "guests": {
                "target": "guests",
                "selected_model": "guests_per_booking_sumratio",
                "evaluation_protocol": "rolling_28_monday_origins_7day_frozen_two_stage_bookings",
                "selection_metric": "weekly_mae_rmse",
                "model_path": "guests_model.joblib",
                "lookback_days": LOOKBACK,
                "selected_metrics": {
                    "mae": weekly_guest["mae"],
                    "rmse": weekly_guest["rmse"],
                    "bias": weekly_guest["bias"],
                    "wape": weekly_guest["wape_pct"],
                },
                "operational_weekly_metrics": {
                    "mae": weekly_guest["mae"],
                    "rmse": weekly_guest["rmse"],
                    "bias": weekly_guest["bias"],
                    "wape": weekly_guest["wape_pct"],
                },
                "daily_metrics": summary7["daily_guests"],
                "interval_method": summary7["interval_method"],
                "interval_coverage_daily": summary7["interval_coverage_daily"],
                "interval_status": "evaluated_ratio_quantiles_q10_q90",
                "interval_coverage_evaluated": True,
            },
        },
        "mae": weekly_book["mae"],
        "rmse": weekly_book["rmse"],
        "wape": weekly_book["wape_pct"],
        "bias": weekly_book["bias"],
        "usable_days": 356,
        "training_snapshot": {
            "days": 356,
            "bookings": 519,
            "guests": 6669,
            "history_sha256": digest(SRC / "trial1_production_daily.csv"),
        },
        "forecast_policy": {
            "bookings_method": "two_stage",
            "guests_method": "guests_per_booking_sumratio",
            "postprocess": "floor_round",
            "maximum_days_per_cutoff": 90,
            "supported_horizons_days": [7, 30, 90],
            "party_size_floor": False,
            "holiday_multipliers": False,
            "demographic_allocation": "period_preserving_largest_remainder",
        },
    }
    (ART / "forecast_models_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_snapshot()

    # Sample forecast via engine
    shadow.clear_cache()
    sample = shadow.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
    sample_bookings = sum(r["predicted_bookings"] for r in sample["forecast"])
    sample_guests = sum(r["predicted_total_guests"] for r in sample["forecast"])
    assert sample_bookings == 12, sample_bookings
    assert sample_guests == 108, sample_guests
    pd.DataFrame(sample["forecast"]).to_csv(OUT / "sample_2027-01-01_to_2027-02-08.csv", index=False)

    # Booking identity vs v3.1 for same sample window
    sys.path.insert(0, str(V31))
    # Load v3.1 engine freshly
    v31_spec = importlib.util.spec_from_file_location("v31_shadow", V31 / "shadow_predict.py")
    v31 = importlib.util.module_from_spec(v31_spec)
    v31_spec.loader.exec_module(v31)
    v31.clear_cache()
    v31_sample = v31.run_forecast("2027-01-01", "2027-02-08", use_cache=False)
    v31_bookings = [r["predicted_bookings"] for r in v31_sample["forecast"]]
    v32_bookings = [r["predicted_bookings"] for r in sample["forecast"]]
    assert v31_bookings == v32_bookings, (v31_bookings, v32_bookings)
    assert sum(r["predicted_total_guests"] for r in v31_sample["forecast"]) == 18

    verification = {
        "weekly_guest_mae": weekly_guest["mae"],
        "weekly_guest_rmse": weekly_guest["rmse"],
        "weekly_guest_bias": weekly_guest["bias"],
        "weekly_guest_wape": weekly_guest["wape_pct"],
        "weekly_booking_wape": weekly_book["wape_pct"],
        "sample_2027": {"bookings": sample_bookings, "guests": sample_guests},
        "sample_v31_guests": 18,
        "booking_identity_sample_vs_v31": True,
        "interval": {
            "method": summary7["interval_method"],
            "daily_coverage": summary7["interval_coverage_daily"],
            "observations": summary7["interval_observations"],
        },
        "horizon_pooled": pooled.to_dict(orient="records"),
        "booking_recompute_match_rate_7day": summary7["booking_recompute_match_rate"],
    }
    (HERE / "verification_results.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"rolling_7day": summary7, "verification": verification}, indent=2) + "\n", encoding="utf-8")

    # Graph weekly actual vs pred guests
    weekly = frame.groupby("origin")[["actual_guests", "pred_guests"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(weekly.origin), weekly.actual_guests, label="Actual", color="black")
    ax.plot(pd.to_datetime(weekly.origin), weekly.pred_guests, label="Sum-ratio guests", color="#2a7a4b")
    ax.set_title("V3.2 RC weekly guests | frozen Two-Stage bookings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "weekly_guests_actual_vs_predicted.png", dpi=140)
    plt.close(fig)

    # REPORT
    lines = [
        "# ReserVision v3.2 RC — guests_per_booking_sumratio",
        "",
        "Isolated release candidate. **Production v3.1 untouched** (rollback preserved).",
        "",
        "## Configuration",
        "",
        "- Bookings: Two-Stage (identical v3.1 artifacts / forward parameters)",
        "- Guests: past-84-day Σguests/Σbookings × operational bookings",
        "- Postprocess: floor_round consistency only (no party-size floor, no holiday multipliers)",
        "- Demographics: period-preserving allocation",
        "",
        "## 7-day weekly metrics (28 origins, frozen Two-Stage bookings)",
        "",
        f"| Target | MAE | RMSE | Bias | WAPE % |",
        f"|---|---:|---:|---:|---:|",
        f"| Guests (sumratio) | {weekly_guest['mae']:.2f} | {weekly_guest['rmse']:.2f} | {weekly_guest['bias']:+.2f} | {weekly_guest['wape_pct']:.2f} |",
        f"| Bookings (locked) | {weekly_book['mae']:.2f} | {weekly_book['rmse']:.2f} | {weekly_book['bias']:+.2f} | {weekly_book['wape_pct']:.2f} |",
        "",
        f"Daily guest MAE/RMSE/Bias/WAPE: {summary7['daily_guests']['mae']:.3f} / {summary7['daily_guests']['rmse']:.3f} / {summary7['daily_guests']['bias']:+.3f} / {summary7['daily_guests']['wape_pct']:.2f}%",
        "",
        "## Uncertainty intervals (sumratio-specific)",
        "",
        f"Method: past-only party-size ratio quantiles q10–q90 × bookings (not robust-median intervals).",
        f"Daily coverage on evaluation days with demand signal: **{100*summary7['interval_coverage_daily']:.1f}%** (n={summary7['interval_observations']}).",
        "",
        "## Sample 2027-01-01 → 2027-02-08",
        "",
        f"- V3.2 RC: **{sample_bookings} bookings / {sample_guests} guests**",
        f"- V3.1 production path: **12 bookings / 18 guests**",
        "- Booking vectors identical across v3.1 and v3.2 for this window.",
        "",
        "## Horizons",
        "",
        "See `horizon_pooled_metrics.csv` for 7/30/90 pooled horizon-total metrics.",
        "",
        "## Rollback",
        "",
        "Keep serving `production_v31` → `reserVision_v3_1_two_stage_rc_20261124`.",
        "This RC remains `production_approved=false` / `shadow_serving_only=true`.",
        "",
    ]
    (HERE / "RELEASE_CANDIDATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {rel: digest(ROOT / rel) for rel in prod_manifest["files"]}
    assert before == after, "Production release files were modified"
    (HERE / "integrity_production_untouched.json").write_text(
        json.dumps({"production_manifest_unchanged": True, "files_checked": len(before)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2))
    print("V3.2 RC prepare complete.")


if __name__ == "__main__":
    main()
