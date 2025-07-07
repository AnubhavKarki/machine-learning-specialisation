# Lab 1

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Helper for NumPy sigmoid
def sigmoid_np(z):
    return 1 / (1 + np.exp(-z))

# ================================
# Linear Regression Neuron Example
# ================================

X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

# Plot data
plt.figure()
plt.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.title("Linear Regression Data")
plt.legend()
plt.show()

# Linear neuron with no activation
linear_layer = Dense(units=1, activation='linear')
_ = linear_layer(X_train[0].reshape(1,1))  # triggers weight init

# Set known weights
set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])

# Predictions
prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b

# Plot predictions
plt.figure()
plt.scatter(X_train, Y_train, color='red', label='Data Points')
plt.plot(X_train, prediction_tf, label='TF Prediction')
plt.plot(X_train, prediction_np, linestyle='--', label='NumPy Prediction')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.title('Linear Neuron Output')
plt.legend()
plt.show()

# ================================
# Logistic Regression Neuron Example
# ================================

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)

# Plot data
plt.figure()
plt.scatter(X_train[Y_train[:,0]==1], Y_train[Y_train[:,0]==1], marker='x', c='red', label='y=1')
plt.scatter(X_train[Y_train[:,0]==0], Y_train[Y_train[:,0]==0], facecolors='none', edgecolors='blue', label='y=0')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Logistic Regression Data")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()

# Logistic neuron
model = Sequential([ Dense(1, input_dim=1, activation='sigmoid', name='L1') ])
_ = model(X_train[0].reshape(1,1))  # trigger init

# Set known weights
set_w = np.array([[2]])
set_b = np.array([-4.5])
model.get_layer('L1').set_weights([set_w, set_b])

# Predictions
pred_tf = model.predict(X_train)
pred_np = sigmoid_np(np.dot(X_train, set_w) + set_b)

# Plot predictions
plt.figure()
plt.plot(X_train, pred_tf, label='TF Sigmoid Output')
plt.plot(X_train, pred_np, linestyle='--', label='NumPy Sigmoid Output')
plt.scatter(X_train, Y_train, color='black', label='True Labels')
plt.xlabel('x')
plt.ylabel('sigmoid output')
plt.title('Logistic Neuron Output')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()
