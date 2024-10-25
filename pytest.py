import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('nairobi_price_ex.csv')
X = data['SIZE'].values
y = data['PRICE'].values


def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error between true and predicted values
    """
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(X, y, m, c, learning_rate=0.0001):
    """
    Compute gradients for slope (m) and intercept (c)
    """
    N = len(X)
    y_pred = m * X + c

    # Gradient for slope (m)
    dm = (-2 / N) * np.sum(X * (y - y_pred))

    # Gradient for intercept (c)
    dc = (-2 / N) * np.sum(y - y_pred)

    #parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    return m, c



m = np.random.randn()
c = np.random.randn()
epochs = 10

print("Initial parameters:")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}\n")

print("Training progress:")
for epoch in range(epochs):
    y_pred = m * X + c

    # Calculate error
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch + 1}, MSE: {error:.4f}")

    # Update parameters using gradient descent
    m, c = gradient_descent(X, y, m, c)

print("\nFinal parameters:")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq ft)')
plt.ylabel('Price')
plt.title('Linear Regression: Office Price vs Size')
plt.legend()
plt.grid(True)
plt.show()

# Predict price for 100 sq ft
size_100 = 100
predicted_price = m * size_100 + c
print(f"\nPredicted price for 100 sq ft: {predicted_price:.2f}")