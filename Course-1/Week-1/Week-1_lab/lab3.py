# Lab - 3
import math, copy
import numpy as np
import matplotlib.pyplot as plt

print("=== Linear Regression with Gradient Descent ===")
x_train = np.array([1.0, 2.0])  # House sizes (1000 sqft)
y_train = np.array([300.0, 500.0])  # Prices ($1000s)

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return (1/(2*m)) * cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        err = f_wb - y[i]
        dj_dw += err * x[i]
        dj_db += err
    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    print(f"\nRunning gradient descent with α={alpha} for {num_iters} iterations...")
    w, b = w_in, b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i % math.ceil(num_iters/10) == 0 or i == num_iters-1:
            cost = compute_cost(x, y, w, b)
            print(f"Iter {i:4}: w = {w:8.4f}, b = {b:8.4f}, Cost = {cost:0.2e}")
    
    return w, b

# Main experiment
w_init, b_init = 0, 0
iterations = 10000
alpha = 0.01

print("\nStarting parameters: w=0, b=0")
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)
print(f"\nFinal parameters: w={w_final:.4f}, b={b_final:.4f}")

# Predictions
print("\nHouse Price Predictions:")
print(f"1000 sqft: ${(w_final*1.0 + b_final)*1000:,.0f}")
print(f"2000 sqft: ${(w_final*2.0 + b_final)*1000:,.0f}")

# Quick test with large learning rate
print("\nTesting large learning rate (α=0.8) for 10 iterations...")
w_final, b_final = gradient_descent(x_train, y_train, 0, 0, 0.8, 10)