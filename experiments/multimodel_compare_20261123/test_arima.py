"""ARIMA-focused leakage checks. Run with the experiment .venv interpreter."""
from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from experiment import load_frame
from models import arima_forecast, statsmodels_available

CUTOFF = date(2026, 4, 6)


@unittest.skipUnless(statsmodels_available(), "statsmodels not installed in this interpreter")
class ArimaLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_frame()

    def test_arima_ignores_future_actuals(self):
        clean = arima_forecast(self.frame, CUTOFF, "bookings", (1, 0, 1))
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.demand_date >= pd.Timestamp(CUTOFF), "bookings"] = 999
        poisoned_forecast = arima_forecast(poisoned, CUTOFF, "bookings", (1, 0, 1))
        self.assertEqual(clean, poisoned_forecast)

    def test_arima_seven_day_length(self):
        forecast = arima_forecast(self.frame, CUTOFF, "guests", (1, 1, 1))
        self.assertEqual(len(forecast), 7)


if __name__ == "__main__":
    unittest.main()
