# Lab1 - ML Specialization

import numpy as np
import time

print("\n--- 1-D np.dot Tests ---")
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])

c = np.dot(a, b)
print(f"np.dot(a, b) = {c} (scalar)")

c = np.dot(b, a)
print(f"np.dot(b, a) = {c} (scalar)")

print("\n--- Speed Comparison: Vectorized np.dot vs Loop ---")
def my_dot(x, y):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * y[i]
    return total

np.random.seed(1)
a = np.random.rand(10_000_000)
b = np.random.rand(10_000_000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(f"Vectorized np.dot(a, b) = {c:.4f}")
print(f"Duration: {1000*(toc - tic):.2f} ms")

tic = time.time()
c = my_dot(a, b)
toc = time.time()
print(f"Loop my_dot(a, b) = {c:.4f}")
print(f"Duration: {1000*(toc - tic):.2f} ms")

del a, b

print("\n--- Course 1 Example: Vector-Vector Dot Product ---")
X = np.array([[1], [2], [3], [4]])
w = np.array([2])

print(f"Shape of X[1]: {X[1].shape}")
print(f"Shape of w: {w.shape}")

c = np.dot(X[1], w)
print(f"np.dot(X[1], w) = {c} (scalar)")

print("\n--- Matrix Indexing and Slicing ---")
a = np.arange(6).reshape(-1, 2)
print(f"Matrix a (shape {a.shape}):\n{a}")

print(f"\nAccess element a[2,0]: {a[2, 0]} (scalar)")
print(f"Access row a[2]: {a[2]} (1-D vector with shape {a[2].shape})")

a = np.arange(20).reshape(-1, 10)
print(f"\nMatrix a (shape {a.shape}):\n{a}")

print(f"\nSlice a[0, 2:7:1]: {a[0, 2:7:1]} (1-D slice with shape {a[0, 2:7:1].shape})")

print(f"\nSlice a[:, 2:7:1]:\n{a[:, 2:7:1]} (2-D slice with shape {a[:, 2:7:1].shape})")

print(f"\nFull matrix slice a[:, :]:\n{a[:, :]} (shape {a[:, :].shape})")

print(f"\nRow slice a[1, :]: {a[1, :]} (1-D vector with shape {a[1, :].shape})")
print(f"Row slice a[1]: {a[1]} (1-D vector with shape {a[1].shape})")
