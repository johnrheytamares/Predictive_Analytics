import os
import joblib
import pandas as pd
import pymysql
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

DB = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "eduardos",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bookings_model.joblib")

def load_data():
    conn = pymysql.connect(**DB)
    df = pd.read_sql(
        """
        SELECT check_in_date, total_bookings
        FROM bookings_daily
        ORDER BY check_in_date ASC
        """,
        conn
    )
    conn.close()

    df = df.rename(columns={"check_in_date": "ds", "total_bookings": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)

    # remove duplicates just in case
    df = df.drop_duplicates(subset=["ds"]).sort_values("ds")

    return df

def evaluate_model(model, test_df):
    future_test = pd.DataFrame({"ds": test_df["ds"]})
    forecast = model.predict(future_test)

    yhat = np.maximum(0, np.round(forecast["yhat"].values))
    actual = test_df["y"].values

    mae = mean_absolute_error(actual, yhat)
    rmse = float(np.sqrt(mean_squared_error(actual, yhat)))

    return float(mae), float(rmse)

def train_with_split(df, test_days=90):
    if len(df) <= test_days + 30:
        raise SystemExit(f"❌ Not enough rows for split. Need > {test_days + 30}, got {len(df)}")

    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="additive",
    )

    model.fit(train_df)

    mae, rmse = evaluate_model(model, test_df)

    metrics = {"mae": mae, "rmse": rmse, "test_days": int(test_days)}
    return model, metrics

def retrain_full(df):
    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="additive",
    )
    final_model.fit(df)
    return final_model

def save_model(model, metrics):
    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    print("\n✅ FINAL MODEL SAVED:", MODEL_PATH)
    print("📊 Metrics:", metrics)

if __name__ == "__main__":
    df = load_data()

    if df.empty:
        raise SystemExit("❌ bookings_daily has no data.")

    print("Rows:", len(df))
    print("Date range:", df["ds"].min().date(), "to", df["ds"].max().date())
    print(df.head())

    # 1) Train + evaluate on last 90 days
    model, metrics = train_with_split(df, test_days=90)
    print("\n📊 EVALUATION (Last 90 Days)")
    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])

    # 2) Retrain final model on full data
    final_model = retrain_full(df)

    # 3) Save final model + metrics
    save_model(final_model, metrics)