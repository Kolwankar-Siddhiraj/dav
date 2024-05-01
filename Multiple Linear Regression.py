# Step 1: Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Prepare the data
# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 3)  # Generate 100 samples with 3 features
y = 4 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 2] + 5 + np.random.randn(100)  # y = 4*X1 + 3*X2 + 2*X3 + 5 + noise

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Coefficients:", model.coef_)

# Plot the results
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plot the ideal line
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Multiple Linear Regression")
plt.show()
