import pandas as pd
from prophet import Prophet
import pymysql

# ------------------------
# 1. Load Data From MySQL
# ------------------------

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='eduardos'
)

query = "SELECT check_in_date, total_bookings FROM bookings_daily ORDER BY check_in_date ASC"
df = pd.read_sql(query, connection)

# Prepare for Prophet
df = df.rename(columns={
    "check_in_date": "ds",
    "total_bookings": "y"
})

df["ds"] = pd.to_datetime(df["ds"])

# ------------------------
# 2. Train Model
# ------------------------

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(df)

# ------------------------
# 3. Predict 2026 (365 days)
# ------------------------

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

future_predictions = forecast[['ds', 'yhat']].tail(365)

# ------------------------
# 4. Save to MySQL
# ------------------------

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS bookings_forecast (
    forecast_date DATE PRIMARY KEY,
    predicted_bookings INT
)
""")

for _, row in future_predictions.iterrows():
    cursor.execute("""
        INSERT INTO bookings_forecast (forecast_date, predicted_bookings)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE predicted_bookings = VALUES(predicted_bookings)
    """, (row['ds'].date(), int(row['yhat'])))

connection.commit()
connection.close()

print("✅ Forecast for 2026 saved into bookings_forecast table!")