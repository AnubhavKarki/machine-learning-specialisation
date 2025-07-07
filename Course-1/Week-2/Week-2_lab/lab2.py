import copy, math
import numpy as np
import matplotlib.pyplot as plt


print("Libraries imported and display options set.")

# Problem Statement: Housing price prediction
# Training dataset
# Size (sqft), Number of Bedrooms, Number of floors, Age of Home, Price (1000s dollars)
# 2104, 5, 1, 45, 460
# 1416, 3, 2, 40, 232
# 852, 2, 1, 35, 178

# Create X_train (features) and y_train (target values)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print("\nTraining data (X_train and y_train) created.")

# Display the shape and type of X_train and y_train
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# Initialize parameters w (weights) and b (bias)
# These are pre-chosen values near the optimal for demonstration
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

print("\nInitial model parameters (w_init and b_init) set.")
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# Function to predict a single value using a loop (element by element)
def predict_single_loop(x, w, b):
    """
    single predict using linear regression (element-wise)

    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter

    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0] # Get the number of features
    p = 0          # Initialize prediction
    for i in range(n):
        p_i = x[i] * w[i]  # Multiply feature by its corresponding weight
        p = p + p_i        # Accumulate the products
    p = p + b              # Add the bias term
    return p

print("\n'predict_single_loop' function defined.")

# Get the first row from X_train for prediction
x_vec = X_train[0,:]
print(f"\nFirst training example (x_vec): shape {x_vec.shape}, value: {x_vec}")

# Make a prediction using the loop-based function
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"Prediction using predict_single_loop: shape {f_wb.shape}, prediction: {f_wb}")


# Function to predict a single value using vector dot product
def predict(x, w, b):
    """
    single predict using linear regression (vectorized)
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter

    Returns:
      p (scalar):  prediction
    """
    # Perform dot product of x and w, then add bias b
    p = np.dot(x, w) + b
    return p

print("\n'predict' function (vectorized) defined.")

# Get the first row from X_train for prediction (again)
x_vec = X_train[0,:]
print(f"\nFirst training example (x_vec): shape {x_vec.shape}, value: {x_vec}")

# Make a prediction using the vectorized function
f_wb = predict(x_vec,w_init, b_init)
print(f"Prediction using vectorized 'predict': shape {f_wb.shape}, prediction: {f_wb}")

# Function to compute the cost (J)
def compute_cost(X, y, w, b):
    """
    compute cost for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = X.shape[0] # Number of training examples
    cost = 0.0     # Initialize cost

    # Loop through each training example
    for i in range(m):
        # Calculate the model's prediction for the current example
        f_wb_i = np.dot(X[i], w) + b
        # Add the squared error to the total cost
        cost = cost + (f_wb_i - y[i])**2
    # Divide by (2 * m) to get the average squared error
    cost = cost / (2 * m)
    return cost

print("\n'compute_cost' function defined.")

# Compute and display cost using the initial optimal parameters
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'\nCost at optimal w,b: {cost}')

# Function to compute the gradient for linear regression
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
    """
    m,n = X.shape           # m = number of examples, n = number of features
    dj_dw = np.zeros((n,))  # Initialize gradient for weights to zeros
    dj_db = 0.              # Initialize gradient for bias to zero

    # Loop through each training example
    for i in range(m):
        # Calculate the error for the current example
        err = (np.dot(X[i], w) + b) - y[i]
        # Loop through each feature to update dj_dw
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        # Update dj_db (gradient for bias)
        dj_db = dj_db + err
    # Average the gradients over all examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

print("\n'compute_gradient' function defined.")

# Compute and display gradient at initial w, b
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'\ndj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


# Function for Gradient Descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # Array to store cost J at each iteration (for plotting)
    J_history = []
    # Create a deep copy of w_in to avoid modifying the global variable
    w = copy.deepcopy(w_in)
    b = b_in

    # Loop for the specified number of iterations
    for i in range(num_iters):

        # Calculate the gradient (dj_db, dj_dw)
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters (w and b) using the learning rate (alpha) and gradients
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration (if within a certain limit)
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        # Print cost every certain interval (10 times or as many as iterations if < 10)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history # Return final w, b, and the cost history

print("\n'gradient_descent' function defined.")

# Initialize parameters for gradient descent
initial_w = np.zeros_like(w_init) # Start with all weights as zero
initial_b = 0.0                   # Start with bias as zero

# Set gradient descent settings
iterations = 1000 # Number of iterations to run
alpha = 5.0e-7    # Learning rate

print("\nInitializing parameters and settings for gradient descent.")

# Run gradient descent to find optimal w and b
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)

print(f"\nFinal b,w found by gradient descent: {b_final:0.2f},{w_final} ")

# Make predictions using the final w and b and print them against target values
m, _ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

print("\nPlotting cost vs. iteration.")

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:]) # Plot tail of cost history
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

print("\nLab completed!")