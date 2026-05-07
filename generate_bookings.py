import pandas as pd
import random
from datetime import timedelta, date

# -----------------------------
# Config: Date range for bookings
# -----------------------------
start_date = date(2024, 1, 1)
end_date = date(2025, 12, 31)  # covers 2024 and 2025

# Calculate number of days
delta = end_date - start_date

# -----------------------------
# Generate synthetic bookings
# -----------------------------
records = []
for i in range(delta.days + 1):
    d = start_date + timedelta(days=i)

    # base demand (normal weekdays)
    base = random.randint(10, 25)

    # weekends: Friday (4), Saturday (5), Sunday (6)
    if d.weekday() >= 4:
        base += random.randint(10, 30)

    # high season: March to May
    if d.month in [3, 4, 5]:
        base += random.randint(5, 30)

    # Christmas / New Year spike
    if (d.month == 12 and d.day >= 20) or (d.month == 1 and d.day <= 3):
        base += random.randint(20, 50)

    # add some random noise
    bookings = max(0, int(base + random.gauss(0, 8)))

    # save as dictionary
    records.append({
        "check_in_date": d.isoformat(),
        "total_bookings": bookings
    })

# -----------------------------
# Convert to DataFrame
# -----------------------------
df = pd.DataFrame(records)

# Optional: preview first 5 rows
print(df.head())

# -----------------------------
# Save to CSV
# -----------------------------
df.to_csv("synthetic_bookings_2024_2025.csv", index=False)
print("Synthetic bookings CSV generated! Total rows:", len(df))