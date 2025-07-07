# Lab - 2

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot training data: y=0 blue circles, y=1 red crosses
def plot_data(X, y, ax):
    ax.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], c='blue', marker='o', label='y=0')
    ax.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='red', marker='x', label='y=1')
    ax.legend()

# Draw vertical threshold line on sigmoid plot at z=0
def draw_vthresh(ax, x=0):
    ax.axvline(x=x, color='gray', linestyle='--')

# Data (6 points with 2 features)
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

# Plot the data points
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plot_data(X, y, ax)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_title('Training Data')
ax.axis([0, 4, 0, 3.5])
plt.show()

# Plot sigmoid function with vertical threshold line at 0
z = np.arange(-10, 11)
fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.plot(z, sigmoid(z), c='b')
ax.set_title('Sigmoid function')
ax.set_xlabel('z')
ax.set_ylabel('sigmoid(z)')
draw_vthresh(ax, 0)
plt.grid(True)
plt.show()

# Logistic regression parameters
b = -3
w = np.array([1, 1])

# Decision boundary line: w0*x0 + w1*x1 + b = 0 => x1 = (-b - w0*x0)/w1
x0_vals = np.linspace(0, 4, 100)
x1_vals = (-b - w[0]*x0_vals) / w[1]

# Plot decision boundary + data + shading below boundary
fig, ax = plt.subplots(1,1, figsize=(6,5))
ax.plot(x0_vals, x1_vals, 'b-', label='Decision boundary')

# Shade region below decision boundary
ax.fill_between(x0_vals, x1_vals, ax.get_ylim()[0], color='blue', alpha=0.2)

# Plot data points
plot_data(X, y, ax)

ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_title('Logistic Regression Decision Boundary')
ax.axis([0, 4, 0, 3.5])
ax.legend()
plt.show()
