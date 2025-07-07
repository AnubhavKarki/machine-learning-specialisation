# Lab - 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction Function
def predict(x, w, b):
    return sigmoid(np.dot(x, w) + b)

# Loss Function (Log Loss)
def compute_cost(wb, x, y):
    w = wb[0]
    b = wb[1]
    m = x.shape[0]
    z = np.dot(x, w) + b
    f = sigmoid(z)
    cost = -(1/m) * np.sum(y * np.log(f + 1e-15) + (1 - y) * np.log(1 - f + 1e-15))
    return cost

# Gradient for Optimization
def compute_gradient(wb, x, y):
    w = wb[0]
    b = wb[1]
    m = x.shape[0]
    z = np.dot(x, w) + b
    f = sigmoid(z)
    dw = (1/m) * np.dot(x.T, (f - y))
    db = (1/m) * np.sum(f - y)
    return np.array([dw, db])

# Data
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

# Optimize
initial_params = np.array([0., 0.])
res = minimize(fun=compute_cost, x0=initial_params, args=(x_train, y_train),
               method='BFGS', jac=compute_gradient)

w_opt, b_opt = res.x

# Plot
z_vals = np.linspace(-1, 6, 100)
f_vals = sigmoid(z_vals * w_opt + b_opt)

plt.figure(figsize=(8,4))
plt.plot(z_vals, f_vals, label='Logistic Regression Curve')
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.axhline(0.5, color='grey', linestyle='--', label='Threshold = 0.5')
plt.title("Logistic Regression Fit")
plt.xlabel("x")
plt.ylabel("Predicted Probability")
plt.legend()
plt.grid(True)
plt.show()
