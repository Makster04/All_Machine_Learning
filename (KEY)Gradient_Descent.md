
#### **Key Terms & Definitions**  

1. **Cost Function (Loss Function)** – A mathematical function that measures the error between predicted values and actual values. Common examples include Mean Squared Error (MSE) for regression and Cross-Entropy Loss for classification.  

2. **Gradient** – A vector that represents the slope (derivative) of the cost function concerning each parameter. It indicates the direction and magnitude of the steepest ascent.  

3. **Learning Rate (α or η)** – A hyperparameter that controls the step size in updating model parameters. A small learning rate leads to slow convergence, while a large learning rate may overshoot the minimum.  

4. **Types of Gradient Descent**:  
   - **Batch Gradient Descent (BGD)** – Computes the gradient using the entire dataset before updating the parameters.  
   - **Stochastic Gradient Descent (SGD)** – Updates parameters after computing the gradient for each training example, introducing randomness.  
   - **Mini-Batch Gradient Descent** – A compromise between BGD and SGD, where gradients are computed for small batches of data.  

5. **Convergence** – When gradient descent reaches an optimal solution (a local or global minimum of the cost function).  

6. **Vanishing & Exploding Gradient** – Problems that arise in deep networks where gradients become too small (vanishing) or too large (exploding), causing slow training or instability.  

#### **Coding Example: Gradient Descent for Linear Regression**  
Below is a simple Python implementation of Gradient Descent for a Linear Regression problem:  

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.1, iterations=1000):
    m = len(y)  # Number of samples
    theta = np.random.randn(2, 1)  # Initialize parameters randomly
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (x0 = 1)

    for iteration in range(iterations):
        gradients = (2/m) * X_b.T @ (X_b @ theta - y)  # Compute gradients
        theta -= learning_rate * gradients  # Update parameters

    return theta

# Train the model
theta_final = gradient_descent(X, y)

# Predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta_final

# Plot results
plt.scatter(X, y, color="blue", label="Training Data")
plt.plot(X_new, y_predict, color="red", label="Linear Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print("Estimated coefficients (theta):", theta_final)
```

#### **Summary**  
- Gradient Descent is a fundamental optimization algorithm in machine learning.  
- The choice of learning rate and gradient descent type affects convergence.  
- The algorithm iteratively updates model parameters to minimize the cost function.  
- It is widely used in regression, classification, and deep learning models.  

Would you like an example with Stochastic Gradient Descent (SGD) or Mini-Batch Gradient Descent? 🚀
