# Lab-3

import numpy as np
import matplotlib.pyplot as plt

# --- Simulate data ---
np.random.seed(42)
m = 100  # samples
n = 4    # features

# Simulate features with different scales:
X_orig = np.zeros((m,n))
X_orig[:,0] = np.random.uniform(500, 3500, m)  # size(sqft), large scale
X_orig[:,1] = np.random.randint(1, 5, m)      # bedrooms, small scale
X_orig[:,2] = np.random.randint(1, 3, m)      # floors, small scale
X_orig[:,3] = np.random.uniform(0, 100, m)    # age, medium scale

# Simulate a target y with some noise
true_w = np.array([300, 5000, -10000, -200])  # made-up coefficients
true_b = 50000
noise = np.random.normal(0, 10000, m)
y = X_orig @ true_w + true_b + noise

print(f"Simulated raw feature sample (first 3 rows):\n{X_orig[:3]}")
print(f"Simulated target sample (first 3):\n{y[:3]}")

# --- Z-score normalization function ---
def zscore_normalize_features(X):
    print("\nNormalizing features with z-score normalization...")
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    print(f"Feature means: {mu}")
    print(f"Feature std devs: {sigma}")
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# --- Gradient descent function for linear regression ---
def run_gradient_descent(X, y, num_iters, alpha):
    print(f"\nRunning gradient descent with alpha={alpha}, iterations={num_iters}...")
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    J_history = []

    for i in range(num_iters):
        predictions = X @ w + b
        errors = predictions - y
        J = (1/(2*m)) * np.sum(errors**2)
        J_history.append(J)

        # Gradient calculations
        dw = (1/m) * (X.T @ errors)
        db = (1/m) * np.sum(errors)

        # Parameter update
        w -= alpha * dw
        b -= alpha * db

        # Print progress every 100 iterations
        if i % 100 == 0 or i == num_iters - 1:
            print(f"Iteration {i+1}/{num_iters} - Cost: {J:.2f}")

    return w, b, J_history

# --- Plotting helper for distributions ---
def norm_plot(ax, data, bins=20):
    ax.hist(data, bins=bins, color='skyblue', edgecolor='black')

# --- Normalize features ---
X_norm, X_mu, X_sigma = zscore_normalize_features(X_orig)

# --- Compare peak-to-peak ranges ---
print("\nPeak to Peak ranges by feature:")
print("Raw X:", np.ptp(X_orig, axis=0))
print("Normalized X:", np.ptp(X_norm, axis=0))

# --- Run gradient descent on normalized data ---
w_norm, b_norm, hist = run_gradient_descent(X_norm, y, 1000, 0.1)

# --- Plot cost function convergence ---
plt.figure(figsize=(8,4))
plt.plot(hist)
plt.title("Cost function convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.show()

# --- Plot distributions before and after normalization ---
fig, ax = plt.subplots(1, 2, figsize=(12,4))
for i in range(n):
    norm_plot(ax[0], X_orig[:,i])
ax[0].set_title("Feature distributions before normalization")

for i in range(n):
    norm_plot(ax[1], X_norm[:,i])
ax[1].set_title("Feature distributions after normalization")
plt.show()

# --- Predict on a new example ---
x_house = np.array([1200, 3, 1, 40])
print(f"\nExample house features (raw): {x_house}")

# Normalize example with training mu and sigma
x_house_norm = (x_house - X_mu) / X_sigma
print(f"Normalized example features: {x_house_norm}")

# Prediction
predicted_price = np.dot(x_house_norm, w_norm) + b_norm
print(f"Predicted price (in dollars): ${predicted_price:,.0f}")

# --- Plot prediction vs targets for first feature ---
plt.scatter(X_orig[:,0], y, label="Actual price")
plt.scatter(X_orig[:,0], X_norm @ w_norm + b_norm, label="Predicted price", alpha=0.6)
plt.xlabel("Size (sqft)")
plt.ylabel("Price")
plt.title("Predicted vs Actual Price based on size")
plt.legend()
plt.show()
