# Lab - 3

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plotting function
def plot_data(X, y, ax):
    for i in range(len(y)):
        if y[i] == 1:
            ax.plot(X[i, 0], X[i, 1], 'rx')  # red cross for label 1
        else:
            ax.plot(X[i, 0], X[i, 1], 'bo')  # blue circle for label 0

X_train = np.array([[0.5, 1.5],
                    [1.0, 1.0],
                    [1.5, 0.5],
                    [3.0, 0.5],
                    [2.0, 2.0],
                    [1.0, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.title("Training Data")
plt.show()

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

# Test the cost function
w_tmp = np.array([1, 1])
b_tmp = -3
print("Cost:", compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

x0 = np.arange(0, 6)
x1_b3 = 3 - x0
x1_b4 = 4 - x0

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x0, x1_b3, c="blue", label="b = -3")
ax.plot(x0, x1_b4, c="magenta", label="b = -4")
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.legend(loc="upper right")
plt.title("Decision Boundaries")
plt.show()

w_array = np.array([1, 1])
b_1 = -3
b_2 = -4

print("Cost for b = -3 :", compute_cost_logistic(X_train, y_train, w_array, b_1))
print("Cost for b = -4 :", compute_cost_logistic(X_train, y_train, w_array, b_2))
# Expected:
# Cost for b = -3 : 0.3668667864055175
# Cost for b = -4 : 0.5036808636748461