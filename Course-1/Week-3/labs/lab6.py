import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=8)

# Sigmoid function replacement
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Simple overfit example replacement
class SimpleOverfitExample:
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12,4))
        
    def fit_and_plot(self, degree=6, lambda_=0):
        np.random.seed(0)
        X = np.linspace(0, 1, 20)
        y = np.sin(2 * np.pi * X) + np.random.randn(20) * 0.3
        
        X_poly = np.vstack([X**i for i in range(degree+1)]).T
        
        reg_matrix = lambda_ * np.eye(degree+1)
        reg_matrix[0,0] = 0  # bias not regularized
        
        w = np.linalg.inv(X_poly.T @ X_poly + reg_matrix) @ X_poly.T @ y
        
        X_test = np.linspace(0,1,100)
        X_test_poly = np.vstack([X_test**i for i in range(degree+1)]).T
        y_pred = X_test_poly @ w
        
        ax = self.axs[0 if lambda_ == 0 else 1]
        ax.clear()
        ax.scatter(X, y, color='blue', label='Training Data')
        ax.plot(X_test, y_pred, color='red', label=f'Degree {degree} Fit, λ={lambda_}')
        ax.legend()
        ax.set_title('No Regularization' if lambda_==0 else 'With Regularization')
        ax.grid(True)
        
    def show(self):
        plt.show()

output = SimpleOverfitExample()

# Your functions for cost and gradient remain exactly the same
# (with compute_cost_linear_reg, compute_cost_logistic_reg, compute_gradient_linear_reg, compute_gradient_logistic_reg)

# Run example usage below...

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost = cost / (2 * m)
 
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    
    total_cost = cost + reg_cost
    return total_cost

cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print("Regularized cost (linear):", cost_tmp)

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost/m

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    
    total_cost = cost + reg_cost
    return total_cost

cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print("Regularized cost (logistic):", cost_tmp)

# Use the overfit example visualization
output.fit_and_plot(degree=6, lambda_=0)
output.fit_and_plot(degree=6, lambda_=1)
output.show()
