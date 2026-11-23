# Phase 3 ARIMA Evaluation

## Environment

- Virtual environment: `experiments/multimodel_compare_20261123/.venv`
- Created because no reusable isolated venv existed
- Production `requirements.txt` was **not** modified

Exact commands:

```powershell
cd "C:\Users\John Rhey Tamares\Predictive_Analytics\experiments\multimodel_compare_20261123"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "statsmodels==0.14.4" "pandas==2.2.3" "numpy==2.2.6" "scikit-learn==1.6.1" "scipy==1.15.2"
.\.venv\Scripts\python.exe -m unittest test_arima.py -v
.\.venv\Scripts\python.exe arima_experiment.py
```

Installed versions: statsmodels 0.14.4, pandas 2.2.3, scikit-learn 1.6.1

## Development results (24 weeks)

| Method | Bookings weekly WAPE | Guests weekly WAPE | Bookings daily WAPE | Guests daily WAPE |
|---|---:|---:|---:|---:|
| arima_100 | 42.47 | 56.68 | 74.61 | 118.79 |
| arima_101 | 42.57 | 56.39 | 74.66 | 119.00 |
| arima_111 | 29.12 | **40.25** | 69.00 | 107.34 |
| arima_211 | **28.59** | 41.16 | 68.11 | 108.95 |

Selected on development weekly WAPE only:

- Bookings → `arima_211`
- Guests → `arima_111`

## Reserved confirmatory (8 previously inspected weeks)

| Method | Bookings weekly WAPE | Guests weekly WAPE |
|---|---:|---:|
| arima_211 | 55.82 | 97.03 |
| arima_111 | 56.21 | 96.33 |
| arima_selected_pair | 55.82 | 96.33 |

## Comparison with prior Phase 3 models (development weekly WAPE)

| Family / method | Bookings | Guests |
|---|---:|---:|
| weekday_baseline | 28.76 | **33.77** |
| ridge_alpha_100 | 28.95 | 38.78 |
| random_forest | 30.37 | 40.49 |
| arima_211 | **28.59** | 41.16 |
| arima_111 | 29.12 | 40.25 |

Notes:

- ARIMA(2,1,1) is slightly better than baseline on **bookings weekly WAPE** (28.59 vs 28.76) but worse on daily WAPE (68.11 vs 61.02).
- No ARIMA order beats baseline on **guests**.
- This is not evidence to replace the production weekday baseline.

## Tests

`.\.venv\Scripts\python.exe -m unittest test_arima.py -v` → 2/2 passed (future-actual leakage blocked; 7-day length).

Serving hashes unchanged.
