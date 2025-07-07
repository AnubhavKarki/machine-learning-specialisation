# Lab 3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Normalization

# Simulated Dataset (Temperature °C and Duration min)
np.random.seed(0)
temperature = np.random.uniform(175, 260, 100)
duration = np.random.uniform(12, 17, 100)
labels = np.where((temperature > 190) & (temperature < 240) & (duration > 12.5) & (duration < 15.5), 1, 0)
X = np.column_stack((temperature, duration))
Y = labels.reshape(-1, 1)

# Visualize Data
def plot_data(X, Y):
    plt.figure(figsize=(6, 4))
    plt.scatter(X[Y[:, 0]==1][:, 0], X[Y[:, 0]==1][:, 1], c='green', label='y=1')
    plt.scatter(X[Y[:, 0]==0][:, 0], X[Y[:, 0]==0][:, 1], edgecolors='red', facecolors='none', label='y=0')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Duration (min)")
    plt.title("Coffee Roasting Data")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_data(X, Y)

# Normalize Input
norm_layer = Normalization(axis=-1)
norm_layer.adapt(X)
Xn = norm_layer(X)

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dense layer: one forward pass
def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out

# 2-layer neural network
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2

# Use previously trained weights
W1 = np.array([[-8.93,  0.29, 12.9], [-0.1, -7.32, 10.81]])
b1 = np.array([-9.82, -9.28,  0.96])
W2 = np.array([[-31.18], [-27.59], [-32.56]])
b2 = np.array([15.41])

# Predict for a dataset
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)
    return p

# Test Predictions
X_test = np.array([[200, 13.9], [200, 17]])
X_testn = norm_layer(X_test)
preds = my_predict(X_testn, W1, b1, W2, b2)
yhat = (preds >= 0.5).astype(int)

print("Probabilities:\n", preds)
print("Decisions:\n", yhat)
