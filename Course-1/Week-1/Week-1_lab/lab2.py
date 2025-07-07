# Lab 2

import numpy as np
import matplotlib.pyplot as plt

"""Problem Statement¶
You would like a model which can predict housing prices given the size of the house. Let's use the same two data points as before the previous
lab- a house with 1000 square feet sold for $300,000 and a house with 2000 square feet sold for $500,000.
"""

x_train = np.array([1.0,2.0])       #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])  #(price in 1000s of dollars)

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost