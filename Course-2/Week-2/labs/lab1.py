# Lab 1

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

# Style
plt.style.use('seaborn-darkgrid')

# Softmax function (standalone)
def my_softmax(z):
    ez = np.exp(z - np.max(z))  # for numerical stability
    sm = ez / np.sum(ez)
    return sm

# Visualize softmax output given dynamic z values
def plot_softmax_interactive():
    from matplotlib.widgets import Slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.4)

    z = np.array([1.0, 2.0, 3.0])
    sm = my_softmax(z)
    bars = ax.bar(range(len(z)), sm)
    ax.set_ylim(0, 1)

    # Slider setup
    ax_z0 = plt.axes([0.25, 0.3, 0.65, 0.03])
    ax_z1 = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_z2 = plt.axes([0.25, 0.2, 0.65, 0.03])

    s_z0 = Slider(ax_z0, 'z0', -10, 10, valinit=1.0)
    s_z1 = Slider(ax_z1, 'z1', -10, 10, valinit=2.0)
    s_z2 = Slider(ax_z2, 'z2', -10, 10, valinit=3.0)

    def update(val):
        z = np.array([s_z0.val, s_z1.val, s_z2.val])
        sm = my_softmax(z)
        for i in range(len(sm)):
            bars[i].set_height(sm[i])
        fig.canvas.draw_idle()

    s_z0.on_changed(update)
    s_z1.on_changed(update)
    s_z2.on_changed(update)

    plt.show()

# plot_softmax_interactive()  # Uncomment this line to run the widget plot

# Create synthetic dataset for softmax classification
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)

# Model using softmax directly in output layer (non-preferred)
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

model.fit(X_train, y_train, epochs=10, verbose=0)

# Predict using non-preferred setup
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred[:2])
print("Softmax output range:", np.min(p_nonpreferred), "to", np.max(p_nonpreferred))

# Preferred version (logits in final layer)
preferred_model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(4, activation='linear')  # logits output
])

preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

preferred_model.fit(X_train, y_train, epochs=10, verbose=0)

p_preferred = preferred_model.predict(X_train)
print("Logits (pre-softmax):\n", p_preferred[:2])

# Convert logits to probabilities manually
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print("After softmax:\n", sm_preferred[:2])

# Getting prediction categories
for i in range(5):
    print(f"Logits: {p_preferred[i]}, Predicted Class: {np.argmax(p_preferred[i])}")
