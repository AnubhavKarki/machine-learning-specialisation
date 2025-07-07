# Lab - 4

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# === Helper Functions ===

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log(1 - f_wb_i)
    return cost / m

def compute_gradient_logistic(X, y, w, b): 
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i  = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]
        dj_db += err_i

    dj_dw /= m
    dj_db /= m
        
    return dj_db, dj_dw  

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost_logistic(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")

    return w, b, J_history

def plot_data(X, y, ax):
    pos = y == 1
    neg = y == 0
    ax.plot(X[pos, 0], X[pos, 1], 'rx', label='y=1')
    ax.plot(X[neg, 0], X[neg, 1], 'bo', label='y=0')
    ax.legend()

def plt_prob(ax, w, b):
    x0_vals = np.linspace(0, 4, 100)
    x1_vals = np.linspace(0, 3.5, 100)
    xx0, xx1 = np.meshgrid(x0_vals, x1_vals)
    grid = np.c_[xx0.ravel(), xx1.ravel()]
    probs = sigmoid(np.dot(grid, w) + b).reshape(xx0.shape)
    ax.contourf(xx0, xx1, probs, 25, cmap="RdBu", alpha=0.6)

# === Dataset ===

X_train = np.array([[0.5, 1.5],
                    [1, 1],
                    [1.5, 0.5],
                    [3, 0.5],
                    [2, 2],
                    [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# === Run Gradient Descent ===

w_init = np.zeros(X_train.shape[1])
b_init = 0.
alpha = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_init, b_init, alpha, iters)
print(f"\nUpdated Parameters: w = {w_out}, b = {b_out}")

# === Plot ===

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt_prob(ax, w_out, b_out)
plot_data(X_train, y_train, ax)

ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.axis([0, 4, 0, 3.5])

# Plot decision boundary
x0 = -b_out / w_out[0]
x1 = -b_out / w_out[1]
ax.plot([0, x0], [x1, 0], c='blue', lw=2)

plt.title("Logistic Regression Decision Boundary")
plt.show()

# === Contour Plot for 1D Dataset ===

def plot_logistic_contour(X, y, w_range, b_range):
    w_vals = np.linspace(w_range[0], w_range[1], 100)
    b_vals = np.linspace(b_range[0], b_range[1], 100)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = np.array([W[i, j]])
            b = B[i, j]
            Z[i, j] = compute_cost_logistic(X.reshape(-1,1), y, w, b)

    fig, ax = plt.subplots(1,1, figsize=(6,5))
    CS = ax.contour(W, B, Z, levels=np.logspace(-1, 1.5, 20), cmap='viridis')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("Cost Contour Plot (Logistic Regression)")
    plt.show()

# === 1D Dataset ===

x1_train = np.array([0., 1, 2, 3, 4, 5])
y1_train = np.array([0,  0, 0, 1, 1, 1])

plot_logistic_contour(x1_train, y1_train, w_range=[-1, 7], b_range=[-14, 1])
