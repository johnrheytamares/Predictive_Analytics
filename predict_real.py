import sys
import os
import joblib
import pandas as pd
import numpy as np
import json

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bookings_model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_real_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle.get("metrics", {})

def predict_dates(model, dates):
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    fc = model.predict(future)
    yhat = np.maximum(0, np.round(fc["yhat"].values)).astype(int)
    return yhat.tolist()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    model, metrics = load_model()

    if mode == "date":
        d = sys.argv[2]
        pred = predict_dates(model, [d])[0]
        print(json.dumps({"date": d, "predicted_bookings": pred, "metrics": metrics}))
    elif mode == "range":
        start = sys.argv[2]
        end = sys.argv[3]
        dates = pd.date_range(start=start, end=end, freq="D").strftime("%Y-%m-%d").tolist()
        preds = predict_dates(model, dates)
        rows = [{"date": dates[i], "predicted_bookings": preds[i]} for i in range(len(dates))]
        print(json.dumps({"rows": rows, "metrics": metrics}))
    else:
        print("ERROR: use 'date YYYY-MM-DD' or 'range YYYY-MM-DD YYYY-MM-DD'")
        sys.exit(1)