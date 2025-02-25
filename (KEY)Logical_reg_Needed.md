### **From Linear Regression to Logistic Regression**  

Linear regression is a fundamental technique for modeling relationships between continuous variables. However, when predicting categorical outcomes (e.g., whether an event occurs or not), linear regression is not suitable. Logistic regression addresses this by modeling probabilities instead of direct numerical outputs.

---

## **Why Do We Need Logistic Regression?**  

Linear regression assumes the dependent variable (\( Y \)) is continuous, but in many real-world problems, outcomes are binary (e.g., success/failure, spam/non-spam, disease/no disease). Applying linear regression directly to binary classification leads to:  

1. **Unbounded Predictions**: Linear regression can predict values outside the range \([0,1]\), making probability interpretation meaningless.
2. **Violation of Homoscedasticity**: The variance of errors in binary classification is not constant, contradicting linear regression assumptions.
3. **Inefficient Decision Boundaries**: A straight-line decision boundary may not capture the actual data distribution well.

Logistic regression overcomes these issues by using the **sigmoid function** to constrain outputs between 0 and 1.

---

## **Key Mathematical Concepts**  

### **1. Sigmoid Function (Logistic Function)**
The sigmoid function maps any real number to a range between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \( z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n \) (similar to linear regression). The output represents the probability of the positive class (e.g., probability of an event occurring).

---

### **2. Logistic Regression Model**
\[
P(Y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
\]

where:  
- \( P(Y = 1 | X) \) is the probability that the outcome is **1** given inputs \( X \).  
- \( \beta_0 \) (intercept) and \( \beta_i \) (coefficients) are estimated using **Maximum Likelihood Estimation (MLE)** instead of **Least Squares** (used in linear regression).  
- The **log-odds (logit transformation)** is defined as:

\[
\log \left( \frac{P}{1 - P} \right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
\]

This transformation ensures a **linear relationship** between predictors and log-odds instead of probabilities.

---

### **3. Interpreting Logistic Regression Parameters**  

- **\( \beta_0 \) (Intercept)**: The log-odds of the event occurring when all \( X \)'s are 0.
- **\( \beta_i \) (Coefficients)**: Represents how a unit change in \( x_i \) affects the log-odds of \( Y \).
- **Odds Ratio \( e^{\beta_i} \)**: Measures how a unit increase in \( x_i \) affects the odds of \( Y = 1 \).  
  - \( e^{\beta_i} > 1 \): Increases the odds of the event.
  - \( e^{\beta_i} < 1 \): Decreases the odds of the event.

---

### **4. Cost Function & Optimization**
Unlike linear regression (which uses **Mean Squared Error**), logistic regression uses **Log-Loss (Binary Cross-Entropy)**:

\[
\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log (\hat{y_i}) + (1 - y_i) \log (1 - \hat{y_i}) \right]
\]

Since the function is non-convex with respect to \( \beta \), we optimize using **Gradient Descent**.

---

## **Python Code for Linear to Logistic Regression**
### **1. Using Linear Regression on a Binary Classification Task (Incorrect Approach)**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulated binary classification dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # Binary target

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predictions
X_test = np.linspace(0, 11, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plotting
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X_test, y_pred, label="Linear Regression", color='blue')
plt.xlabel("Feature")
plt.ylabel("Probability")
plt.legend()
plt.show()
```
📌 **Problem**: Predictions are not bounded between [0,1].

---

### **2. Implementing Logistic Regression (Correct Approach)**
```python
from sklearn.linear_model import LogisticRegression

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X, y)

# Predictions (Probability)
y_prob = log_model.predict_proba(X_test)[:, 1]

# Plotting
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X_test, y_prob, label="Logistic Regression", color='green')
plt.xlabel("Feature")
plt.ylabel("Probability")
plt.legend()
plt.show()
```
✅ **Solution**: Outputs are now between **0 and 1**, making them interpretable as probabilities.

---

## **Key Takeaways**
1. **Linear regression is unsuitable for classification** due to unbounded predictions.
2. **Logistic regression models probabilities** using the sigmoid function.
3. **Logit transformation makes the relationship linear** with respect to log-odds.
4. **Model parameters are interpreted as odds ratios**, which indicate how predictors influence the likelihood of an event occurring.
5. **Gradient Descent is used for optimization**, as MLE does not have a closed-form solution.

Would you like me to explain anything further, like **regularization in logistic regression** or **multiclass extensions (Softmax Regression)?** 🚀
