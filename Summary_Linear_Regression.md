# Comprehensive Guide to Linear Regression

## 1. Introduction to Linear Regression
Linear Regression is a fundamental statistical and machine learning technique used to model relationships between dependent and independent variables. It is used for predictive modeling and trend analysis.

---

## 2. Types of Linear Regression
### a. Simple Linear Regression
- Involves one independent variable and one dependent variable.
- The relationship is modeled with a straight line.
- Equation:

  $$\[ y = \beta_0 + \beta_1 x + \epsilon \]$$

### b. Multiple Linear Regression
- Involves two or more independent variables.
- Equation:

  $$\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon \]$$

Where:

- $$\( y \)$$ **(Dependent Variable)**:  
  The target or outcome variable that we are trying to predict based on the independent variables. It depends on the values of $$\( x_i \)$$.  

- $$\( x_i \)$$ **(Independent Variables / Predictors)**:  
  The input features or explanatory variables used to predict $$\( y \)$$. These are the variables that influence the dependent variable.  

- $$\( \beta_0 \)$$ **(Intercept)**:  
  The constant term in the regression equation, representing the predicted value of $$\( y \)$$ when all independent variables are zero.  

- $$\( \beta_i \)$$ **(Coefficients / Weights)**:  
  The parameters that determine the influence of each independent variable on the dependent variable. Each $$\( \beta_i \)$$ represents the change in $$\( y \)$$ for a one-unit change in $$\( x_i \)$$, assuming all other variables remain constant.  

- $$\( \epsilon \)$$ **(Error Term / Residuals)**:  
  The difference between the actual observed values of $$\( y \)$$ and the predicted values. It accounts for variability in $$\( y \)$$ that is not explained by the independent variables.  


---

## 3. Objective of Linear Regression
The goal is to minimize the difference between predicted and actual values, which is achieved by minimizing the **Sum of Squared Errors (SSE):**

$$\[ SSE = \sum (y_i - \hat{y}_i)^2 \]$$

Where:
- $$\( y_i \)$$ = Actual observed value
- $$\( \hat{y}_i \)$$ = Predicted value

---

## 4. Assumptions of Linear Regression
1. **Linearity**: The relationship between the dependent and independent variables must be linear.
2. **Homoscedasticity**: The variance of residuals should remain constant across all levels of the independent variables.
3. **Normality of Errors**: The residuals (errors) should be normally distributed.
4. **No Multicollinearity**: Independent variables should not be highly correlated with each other.
5. **Independence of Errors**: Errors should be independent of each other (no autocorrelation).

---

## 5. Evaluation Metrics for Linear Regression
- **Mean Absolute Error (MAE):**

  $$\[ MAE = \frac{1}{n} \sum |y_i - \hat{y}_i| \]$$
  
- **Mean Squared Error (MSE):**

  $$\[ MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \]$$

- **Root Mean Squared Error (RMSE):**

  $$\[ RMSE = \sqrt{MSE} \]$$

- **R-Squared $$(\( R^2 \))$$ Coefficient:**

  $$\[ R^2 = 1 - \frac{SS_{residual}}{SS_{total}} \]$$

Where:
- $$\( SS_{residual} \)$$ = Sum of squared errors
- $$\( SS_{total} \)$$ = Total sum of squares

---

## 6. Regularization in Linear Regression
Regularization helps prevent overfitting by adding penalty terms to the loss function.

### a. Ridge Regression (L2 Regularization)
- Adds the sum of squared coefficients to the loss function:
  $$\[ Loss = SSE + \lambda \sum \beta_i^2 \]$$

### b. Lasso Regression (L1 Regularization)
- Adds the sum of absolute values of coefficients to the loss function:
  $$\[ Loss = SSE + \lambda \sum |\beta_i| \]$$
- Lasso can shrink some coefficients to zero, effectively selecting important features.

---

## 7. Polynomial Regression
- Extends linear regression by fitting a polynomial function to the data.
- Equation:

  $$\[ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n + \epsilon \]$$
  
- Useful for capturing non-linear relationships.

---

## 8. Python Implementation of Linear Regression
### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
```
### Step 2: Load and Prepare Data
```python
# Sample dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 32000, 34000, 36000, 40000, 43000, 46000, 50000, 52000, 55000]
}
df = pd.DataFrame(data)

# Split data
X = df[['Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Step 3: Fit the Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
### Step 4: Make Predictions
```python
y_pred = model.predict(X_test)
```
### Step 5: Evaluate the Model
```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
```
### Step 6: Visualize Results
```python
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Fitted Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Salary Prediction")
plt.legend()
plt.show()
```
```(101813.31747919036, 0.9989818668252081)```
![image](https://github.com/user-attachments/assets/ece9468c-487a-4202-b0b9-bd7f97e6febd)


---

## 9. Conclusion
Linear Regression is a fundamental statistical tool used in predictive modeling. Understanding the underlying assumptions, evaluation metrics, and advanced concepts such as regularization and polynomial regression allows for better model performance and interpretation.

### When to Use Each Method:
| Method | Best Use Case |
|--------|-------------|
| **Multiple Linear Regression** | When the relationship between variables is linear. |
| **Polynomial Regression** | When data exhibits a non-linear trend. |
| **Ridge/Lasso Regression** | When overfitting or multicollinearity is present. |

This guide provides a comprehensive overview of Linear Regression, ensuring a solid foundation in both theoretical and practical aspects of the topic.

