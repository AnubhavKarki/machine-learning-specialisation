# Lab 2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

# Disable TF logs
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# --- Step 1: Simulate Dataset ---
np.random.seed(42)
temperature = np.random.uniform(175, 260, 100)
duration = np.random.uniform(12, 17, 100)
labels = np.where((temperature > 190) & (temperature < 240) & (duration > 12.5) & (duration < 15.5), 1, 0)
X = np.column_stack((temperature, duration))
Y = labels.reshape(-1, 1)

# --- Step 2: Plot Dataset ---
plt.figure(figsize=(6, 4))
plt.scatter(X[Y[:, 0] == 1][:, 0], X[Y[:, 0] == 1][:, 1], c='green', marker='x', label='Good Roast (y=1)')
plt.scatter(X[Y[:, 0] == 0][:, 0], X[Y[:, 0] == 0][:, 1], facecolors='none', edgecolors='red', label='Bad Roast (y=0)')
plt.xlabel("Temperature (°C)")
plt.ylabel("Duration (min)")
plt.title("Coffee Roasting Data")
plt.legend()
plt.grid(True)
plt.show()

# --- Step 3: Normalize Data ---
norm_layer = Normalization(axis=-1)
norm_layer.adapt(X)
Xn = norm_layer(X)

# --- Step 4: Duplicate Data ---
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))

# --- Step 5: Build Model ---
tf.random.set_seed(1234)
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation='sigmoid', name='layer1'),
    Dense(1, activation='sigmoid', name='layer2')
])
model.summary()

# --- Step 6: Train ---
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01))
model.fit(Xt, Yt, epochs=10, verbose=1)

# --- Step 7: Weights ---
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("\nW1:\n", W1, "\nb1:", b1)
print("\nW2:\n", W2, "\nb2:", b2)

# --- Step 8: Prediction ---
X_test = np.array([[200, 13.9], [200, 17]])
X_testn = norm_layer(X_test)
preds = model.predict(X_testn)
print("\nPredicted probabilities:\n", preds)
print("Decisions:\n", (preds >= 0.5).astype(int))
