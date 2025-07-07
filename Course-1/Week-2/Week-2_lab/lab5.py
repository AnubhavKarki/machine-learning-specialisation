# Lab - 5

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=2)
plt.style.use('seaborn-v0_8-darkgrid')  # fallback style

# ---- Mock housing data ----
# Features: [size(sqft), bedrooms, floors, age]
X_train = np.array([
    [2104, 3, 1, 45],
    [1416, 2, 2, 40],
    [1534, 3, 2, 30],
    [852, 2, 1, 36]
])
y_train = np.array([460, 232, 315, 178])  # Target: price
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# ---- Normalize features using z-score ----
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

print(f"Peak to Peak range by column in Raw        X: {np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized X: {np.ptp(X_norm, axis=0)}")

# ---- Train model using SGDRegressor ----
sgdr = SGDRegressor(max_iter=1000, tol=1e-3)
sgdr.fit(X_norm, y_train)

print(f"\nSGDRegressor: {sgdr}")
print(f"Number of iterations: {sgdr.n_iter_}, Updates: {sgdr.t_}")

# ---- View parameters ----
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"Model parameters: w: {w_norm}, b: {b_norm}")

# ---- Make predictions ----
y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"Predictions match: {(np.allclose(y_pred, y_pred_sgd))}")
print(f"\nPredictions: {y_pred[:4]}")
print(f"Targets:     {y_train[:4]}")

# ---- Plot predictions vs original features ----
fig, ax = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='Target')
    ax[i].scatter(X_train[:, i], y_pred, color='orange', marker='x', label='Prediction')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target vs Prediction using Z-score Normalized Model")
plt.tight_layout()
plt.show()
