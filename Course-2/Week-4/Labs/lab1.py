# Lab 1

import numpy as np

# Entropy function
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Split indices for a given feature
def split_indices(X, index_feature):
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

# Weighted entropy of the split
def weighted_entropy(X, y, left_indices, right_indices):
    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    p_left = sum(y[left_indices]) / len(left_indices)
    p_right = sum(y[right_indices]) / len(right_indices)
    return w_left * entropy(p_left) + w_right * entropy(p_right)

# Information gain calculation
def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy

# Sample training data (features are already one-hot encoded)
X_train = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

# === Testing Section ===
features = ['Ear Shape', 'Face Shape', 'Whiskers']

print("Information Gain for each feature:\n")
for i, feature_name in enumerate(features):
    left_indices, right_indices = split_indices(X_train, i)
    gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"{feature_name}: {gain:.4f}")
