import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# Z-score normalization function
def zscore_normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

# Gradient descent for linear regression
def run_gradient_descent_feng(X, y, iterations=1000, alpha=0.01):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    y = y.reshape(-1, 1)

    for i in range(iterations):
        y_pred = X @ w + b
        error = y_pred - y
        dw = (1/m) * (X.T @ error)
        db = (1/m) * np.sum(error)
        w -= alpha * dw
        b -= alpha * db

    return w, b

# --- Simple quadratic fit without feature engineering ---
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)

plt.scatter(x, y, c='r', marker='x', label="Actual Value")
plt.plot(x, X @ model_w + model_b, label="Predicted Value")
plt.title("No Feature Engineering")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"w,b found: w={model_w.flatten()}, b={model_b:.4f}\n")

# --- Using x^2 as engineered feature for polynomial regression ---
X_poly = (x**2).reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X_poly, y, iterations=200000, alpha=1e-5)

plt.scatter(x, y, c='r', marker='x', label="Actual Value")
plt.plot(x, X_poly @ model_w + model_b, label="Predicted Value")
plt.title("Feature Engineered with x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"w,b found: w={model_w.flatten()}, b={model_b:.4f}\n")

# --- Multiple polynomial features x, x^2, x^3 ---
X_multi = np.c_[x, x**2, x**3]

model_w, model_b = run_gradient_descent_feng(X_multi, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, c='r', marker='x', label="Actual Value")
plt.plot(x, X_multi @ model_w + model_b, label="Predicted Value")
plt.title("Features: x, x^2, x^3")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"w,b found: w={model_w.flatten()}, b={model_b:.4f}\n")

# --- Visualizing linearity of features with respect to target ---
feature_names = ['x', 'x^2', 'x^3']

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i, ax in enumerate(axes):
    ax.scatter(X_multi[:, i], y)
    ax.set_xlabel(feature_names[i])
axes[0].set_ylabel("y")
plt.suptitle("Feature vs Target Linearity")
plt.show()

# --- Feature scaling with Z-score normalization ---
print(f"Peak-to-peak range before scaling: {np.ptp(X_multi, axis=0)}")
X_scaled = zscore_normalize_features(X_multi)
print(f"Peak-to-peak range after scaling: {np.ptp(X_scaled, axis=0)}\n")

# --- Fit again with normalized features and higher learning rate ---
model_w, model_b = run_gradient_descent_feng(X_scaled, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, c='r', marker='x', label="Actual Value")
plt.plot(x, X_scaled @ model_w + model_b, label="Predicted Value")
plt.title("Normalized Features x, x^2, x^3")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"w,b found: w={model_w.flatten()}, b={model_b:.4f}\n")

# --- Modeling complex function: cos(x/2) with high-degree polynomial features ---
y_complex = np.cos(x / 2)
X_complex = np.c_[x]  # start with x

# Add polynomial features up to degree 13
for deg in range(2, 14):
    X_complex = np.c_[X_complex, x**deg]

X_complex_scaled = zscore_normalize_features(X_complex)

model_w, model_b = run_gradient_descent_feng(X_complex_scaled, y_complex, iterations=1_000_000, alpha=1e-1)

plt.scatter(x, y_complex, c='r', marker='x', label="Actual Value")
plt.plot(x, X_complex_scaled @ model_w + model_b, label="Predicted Value")
plt.title("Modeling cos(x/2) with Polynomial Features")
plt.xlabel("x")
plt.ylabel("cos(x/2)")
plt.legend()
plt.show()

print(f"w,b found: w={model_w.flatten()}, b={model_b:.4f}\n")
