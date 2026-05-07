import pandas as pd

# Load the CSV
df = pd.read_csv("synthetic_bookings_2024_2025.csv")

# Check first few rows
print(df.head())

# Ensure the date column is datetime
df['check_in_date'] = pd.to_datetime(df['check_in_date'])