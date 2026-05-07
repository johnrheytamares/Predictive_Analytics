import pandas as pd

# Step 0: Load your synthetic bookings CSV
df = pd.read_csv("synthetic_bookings_2024_2025.csv")

# Ensure the date column is in datetime format
df['check_in_date'] = pd.to_datetime(df['check_in_date'])

# Step 1: Daily bookings
daily = df.groupby('check_in_date')['total_bookings'].sum().reset_index()
print("Daily bookings:")
print(daily.head())

# Step 2: Weekly bookings
df['week'] = df['check_in_date'].dt.isocalendar().week
df['year'] = df['check_in_date'].dt.year
weekly = df.groupby(['year', 'week'])['total_bookings'].sum().reset_index()
print("\nWeekly bookings:")
print(weekly.head())

# Step 3: Monthly bookings
df['month'] = df['check_in_date'].dt.month
monthly = df.groupby(['year', 'month'])['total_bookings'].sum().reset_index()
print("\nMonthly bookings:")
print(monthly.head())

# Step 4: Yearly bookings
yearly = df.groupby('year')['total_bookings'].sum().reset_index()
print("\nYearly bookings:")
print(yearly)

# Step 5: Save all summaries to CSV (optional but recommended)
daily.to_csv("bookings_daily.csv", index=False)
weekly.to_csv("bookings_weekly.csv", index=False)
monthly.to_csv("bookings_monthly.csv", index=False)
yearly.to_csv("bookings_yearly.csv", index=False)

print("\nAll summaries saved as CSV!")