import os
print("Current folder:", os.getcwd())
print("Files:", os.listdir('.'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load and prepare data
df = pd.read_csv('Nat_Gas.csv')
df['Date'] = pd.to_datetime(df['Dates'])
df = df.sort_values('Date').reset_index(drop=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print("Data preview:")
print(df[['Date', 'Prices']].head())
print(df[['Date', 'Prices']].tail())
print(f"Data shape: {df.shape}")

# Visualize data and patterns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Time series plot
ax1.plot(df['Date'], df['Prices'], marker='o', linewidth=2, markersize=4)
ax1.set_title('Natural Gas Monthly End Prices (Oct 2020 - Sep 2024)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Seasonal pattern (average by month)
monthly_avg = df.groupby('Month')['Prices'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.bar(range(1,13), monthly_avg.values, alpha=0.7, color='skyblue', edgecolor='navy')
ax2.set_title('Average Price by Month (Seasonal Trends)')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Price ($)')
ax2.set_xticks(range(1,13))
ax2.set_xticklabels(months)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nat_gas_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Simple polynomial trend model (no sklearn needed)
# Use polyfit for quadratic trend: price ~ a*year^2 + b*year + c + seasonal
years = df['Year'] - df['Year'].min()  # Normalize years
X = np.column_stack([years**2, years, df['Month']])
y = df['Prices'].values

# Fit quadratic model using numpy polyfit (multiple variables via normal equation)
XtX = np.dot(X.T, X)
XtX_inv = np.linalg.inv(XtX)
Xt_y = np.dot(X.T, y)
coefficients = np.dot(XtX_inv, Xt_y)
print(f"\nModel coefficients: {coefficients}")

def estimate_price(date_str):
    """
    Estimate natural gas price for any date.
    - Past: Interpolates trend + seasonality
    - Future: Extrapolates up to 1+ year
    Returns price rounded to 2 decimals.
    """
    dt = pd.to_datetime(date_str)
    year, month = dt.year, dt.month
    years_norm = year - df['Year'].min()
    X_pred = np.array([years_norm**2, years_norm, month])
    pred = np.dot(X_pred, coefficients)
    return max(8.0, round(pred, 2))  # Floor at ~historical min

# Test cases
test_dates = ['2022-03-31', '2024-09-30', '2023-06-30']
print("\nTest estimates vs actual:")
for date in test_dates:
    pred = estimate_price(date)
    actual = df[df['Date'].dt.strftime('%Y-%m-%d') == pd.to_datetime(date).strftime('%Y-%m-%d')]['Prices'].values
    actual_str = f"${actual[0]:.2f}" if len(actual)>0 else "N/A"
    print(f"{date}: Predicted ${pred}, Actual {actual_str}")

# One-year extrapolation (Oct 2024 - Sep 2025)
print("\nOne-year forward estimates:")
future_start = '2024-10-31'
future_dates = pd.date_range(start=future_start, periods=12, freq='MS').strftime('%Y-%m-%d')
future_prices = [estimate_price(d) for d in future_dates]
for date, price in zip(future_dates, future_prices):
    print(f"{date}: ${price}")

# Model fit metric (R² equivalent)
y_pred = np.dot(X, coefficients)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"\nModel R²: {r2:.3f} (explains {r2*100:.1f}% of variance)")
