import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load Historical Data from CSV
# Provide the path to your CSV file
csv_file_path = "data/tcs_stock_data.csv"  # Update this path with your file location

# Load the data into a pandas DataFrame
data = pd.read_csv(csv_file_path)

# Check if the file is loaded successfully
if data.empty:
    print(f"❌ Failed to load data from {csv_file_path}. Please check the file.")
    exit()

# Display first few rows of the dataset to verify
print(data.head())

# Step 2: Preprocess Data (ensure the 'Close' column exists)
if 'Close' not in data.columns:
    print("❌ 'Close' column not found in the CSV file.")
    exit()

# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()
data = data.dropna()  # Drop missing values

# Step 3: Calculate Mean and Standard Deviation of Returns
mu = data['Daily Return'].mean()
sigma = data['Daily Return'].std()

print(f"✅ Data loaded successfully")
print(f"Mean Daily Return: {mu:.5f}")
print(f"Standard Deviation: {sigma:.5f}")

# Step 4: Monte Carlo Simulation Parameters
simulations = 1000
days = 252  # typical number of trading days in a year
start_price = data['Close'].iloc[-1]  # Last closing price as the starting price

# Step 5: Run Monte Carlo Simulations
simulation_results = np.zeros((simulations, days))

for i in range(simulations):
    daily_returns = np.random.normal(loc=mu, scale=sigma, size=days) + 1
    price_series = [start_price]

    for j in range(1, days):
        price_series.append(price_series[j-1] * daily_returns[j])

    simulation_results[i] = price_series

# Step 6: Plot the Simulations
plt.figure(figsize=(12, 6))
plt.plot(simulation_results.T, alpha=0.05, color='blue')
plt.title('Monte Carlo Simulation: Stock Price Forecast (1 Year)')
plt.xlabel('Day')
plt.ylabel('Price (INR)')
plt.grid(True)
plt.show()

# Step 7: Plot Histogram of Final Prices
ending_prices = simulation_results[:, -1]
plt.figure(figsize=(10, 6))
plt.hist(ending_prices, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Simulated Stock Prices')
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 8: Print Risk Boundaries
percentiles = np.percentile(ending_prices, [5, 50, 95])
print(f"\n📊 Stock Price Forecast Summary after 1 Year:")
print(f"5th Percentile (Bear Case): ₹{percentiles[0]:.2f}")
print(f"Median Price: ₹{percentiles[1]:.2f}")
print(f"95th Percentile (Bull Case): ₹{percentiles[2]:.2f}")
