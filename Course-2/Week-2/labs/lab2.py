# Multi-Class Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# For reproducibility
np.random.seed(30)
tf.random.set_seed(1234)

# Generate 4-class dataset
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=30)

# Plot dataset
colors = ['blue', 'green', 'orange', 'purple']
for i in range(classes):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], c=colors[i], label=f'Class {i}')
plt.title("Training Data")
plt.xlabel("x0")
plt.ylabel("x1")
plt.legend()
plt.grid(True)
plt.show()

# Build model
model = Sequential([
    Dense(2, activation='relu', name="L1"),
    Dense(4, activation='linear', name="L2")
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

# Train model
model.fit(X_train, y_train, epochs=200, verbose=0)

# Plot decision boundaries
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits = model.predict(grid)
    Z = np.argmax(logits, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.get_cmap('Set1', classes))
    for i in range(classes):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', edgecolor='k')
    plt.title("Decision Boundaries")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X_train, y_train)
