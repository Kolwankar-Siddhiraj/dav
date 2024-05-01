# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Generate some time series data
dates = pd.date_range(start='2022-01-01', end='2022-12-31')
data = np.random.randn(len(dates))  # Random data points

# Step 3: Create a pandas DataFrame
time_series = pd.DataFrame(data, index=dates, columns=['Value'])

# Step 4: Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(time_series.index, time_series['Value'], color='blue')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Step 5: Perform basic time series analysis (e.g., rolling mean)
window_size = 30  # Size of the rolling window
time_series['Rolling Mean'] = time_series['Value'].rolling(window=window_size).mean()

# Step 6: Plot the original data and rolling mean
plt.figure(figsize=(10, 6))
plt.plot(time_series.index, time_series['Value'], color='blue', label='Original Data')
plt.plot(time_series.index, time_series['Rolling Mean'], color='red', label='Rolling Mean')
plt.title('Time Series Data with Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()