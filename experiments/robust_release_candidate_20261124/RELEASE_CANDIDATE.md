# Step 2F — Robust Median Production Release Candidate

**Package ID:** `robust_median_rc_20261124`  
**Scope label:** `release-candidate-not-deployed`  
**Production approved:** No  
**Staging approved:** No (awaiting explicit approval)  
**GO/NO-GO for staging:** **GO for staging only** (see §7)

No Dokploy deploy, secret changes, DB access, retraining, or canonical replacement was performed.

---

## 1. Release-candidate directory and inventory

Root: `Predictive_Analytics/experiments/robust_release_candidate_20261124/`

| Path | Role |
|------|------|
| `artifacts/bookings_model.joblib` | Byte-identical canonical weekday baseline |
| `artifacts/guests_model.joblib` | Same history; `model_type=robust_median_guests` only |
| `artifacts/forecast_models_metadata.json` | RC metadata + corrected 90-day `forecast_policy` |
| `artifacts/bookings_model_metadata.json` | Byte-identical canonical companion |
| `prepare_release_candidate.py` | Reproducible builder (refuses overwrite) |
| `verify_fastapi.py` | Isolated FastAPI ASGI tests |
| `source_integrity.json` | Canonical + RC SHA-256 |
| `PACKAGE_MANIFEST.json` | Full inventory + verification summary |
| `verification_fastapi.json` | FastAPI results |
| `verification_node.json` | Node bridge results |
| `RELEASE_CANDIDATE.md` | This document |

Related (backend, not inside Python package):

- `CAPSTONE_BACKEND/reservision-backend/experiments/robust_release_candidate_20261124/verify-rc-local.mjs`

---

## 2. Corrected metadata and forecast policy

| Field | Value |
|-------|-------|
| `deployment_scope` | `release-candidate-not-deployed` |
| `production_approved` | `false` |
| Bookings `selected_model` | `weekday_baseline` |
| Guests `selected_model` | `robust_median_guests` |
| Post-process | Variant A (`typical_guests_per_booking=7.0`) |
| `forecast_policy.profile` | `cutoff_anchored_four_week_weekday_median` |
| `forecast_policy.maximum_days_per_cutoff` | **90** |
| `supported_dates` length | **90** |
| `recursive_predictions` | `false` |
| `fallback` | `null` |
| Interval status | **Not evaluated** (uncalibrated residual bands) |
| Guest holdout metrics | Cleared (not relabeled as robust validation) |

Embedded `forecast_policy` now matches live `/health` capabilities for max days and profile (Step 2E stale max-7 blocker resolved for this candidate).

Algorithm and historical series unchanged: no retraining; guests history equals canonical content; bookings file SHA equals canonical.

---

## 3. SHA-256 manifest

See `PACKAGE_MANIFEST.json` and `source_integrity.json`.

Canonical (unchanged):

| File | SHA-256 |
|------|---------|
| `bookings_model.joblib` | `d629e26d07c98107a62a745dde0cad13156d72da6582af1aa16fa11c30525487` |
| `guests_model.joblib` | `2f0725fb9c1b3207fa9c2acb30126f6b99c222bbcbdd42552faef11a882771ef` |
| `forecast_models_metadata.json` | `60f06f2c7a2d8f89f4bad5fc5ff75e0e79b13234819f8bf6fd1bf6df9953565a` |
| `bookings_model_metadata.json` | `d7d8c03aad381f31d0d47e7afbdc03377ca26f8a79cddda91e8b0047229ebe48` |

RC bookings artifact SHA **equals** canonical bookings SHA.

---

## 4. Verification results

| Suite | Result |
|-------|--------|
| FastAPI RC (`verify_fastapi.py`) | **4/4 passed** |
| Node RC bridge (`verify-rc-local.mjs`) | **Passed** (admin 200; anon 401; customer 403; robust+baseline 7/30/90) |
| Node unit (forecastRobust + predictiveAnalytics) | **11/11 passed** |
| Vue `verify-forecast-capabilities.mjs` | **Passed** |
| Vue `verify-forecast-dashboard.mjs` | **Passed** |
| Failures | **0** |

### Local forecast totals (Variant A)

FastAPI RC start `2026-11-22`:

| Days | Bookings | Guests |
|-----:|---------:|-------:|
| 7 | 2 | 14 |
| 30 | 11 | 77 |
| 90 | 36 | 252 |

Node bridge start `2026-11-25` (RC):

| Days | Bookings | Guests |
|-----:|---------:|-------:|
| 7 | 2 | 14 |
| 30 | 11 | 77 |
| 90 | 37 | 259 |

Canonical baseline path still returns 4/30, 16/119, 76/566 for the same Node start dates (unchanged behavior).

Unsupported insufficient-weekday history still returns `ROBUST_RANGE_UNSUPPORTED` (no silent baseline fallback). Future actuals ignored; weekday profile reused across horizon.

---

## 5. Canonical integrity

Confirmed byte-for-byte unchanged before/after prepare and after FastAPI + Node verification.

Frozen Phase 2–5 evaluation package: not modified.

---

## 6. Staging verification checklist

Use **staging** (or an isolated Dokploy preview), **not** production, only after explicit staging approval.

1. Build Python image from current `predict_real.py` / `api.py` with **RC `artifacts/`** copied to `/app` (do not use Step 2A local-only folder).  
2. Deploy Python → confirm `/health`: scope `release-candidate-not-deployed`; guests `robust_median_guests`; bookings `weekday_baseline`; capabilities max 90; interval Not evaluated.  
3. Point staging Node `PREDICTIVE_ANALYTICS_URL` at that service (no production env edits from this package).  
4. Deploy Step 2D Node + Vue builds to staging.  
5. Admin auth: readiness labels; generate 7/30/90; demographics sum; Variant A floor; structured unsupported error.  
6. Real browser check of range controls and interval wording (uncalibrated / Not evaluated).  
7. Keep rollback inventory below ready before any cutover discussion.

### Rollback artifact inventory (production baseline)

Restore these canonical files / prior Python image digest:

- SHAs in §3 (and `PACKAGE_MANIFEST.json` → `rollback_inventory`)  
- Prior Dokploy predictive-analytics release ID (record at staging start)

---

## 7. Remaining blockers (production)

Still **NO-GO for production**:

1. No written **production** activation approval  
2. Staging must succeed first (checklist §6)  
3. Authenticated real-browser + live MySQL capacity/recommendation checks on staging  
4. Accuracy limitations remain (10–20% weekly WAPE target unmet; intervals uncalibrated)  
5. Package explicitly **not** production-approved (`release-candidate-not-deployed`)

---

## 8. GO / NO-GO

| Target | Recommendation |
|--------|----------------|
| **Staging** | **GO** — artifact/metadata blockers from Step 2E are resolved in this RC; local FastAPI/Node/Vue verification passed |
| **Production** | **NO-GO** — await staging success + explicit production approval |

---

## 9. Stop

Local preparation and verification complete. Wait for explicit approval before any deployment.
