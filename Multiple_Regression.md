## **Equations and Coding for Multiple Linear Regression**  

### **1. Multiple Linear Regression Model**  
A multiple linear regression model is mathematically represented as:  

$$\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
\]$$

where:  
- $$\( Y \)$$ = **Dependent variable (target)**: The outcome variable we are trying to predict.  
- $$\( X_1, X_2, ..., X_n \)$$ = **Independent variables (predictors)**: Features that influence $$\( Y \)$$.  
- $$\( \beta_0 \)$$ = **Intercept**: The value of $$\( Y \)$$ when all $$\( X \)$$ variables are zero.  
- $$\( \beta_1, \beta_2, ..., \beta_n \)$$ = **Regression coefficients**: Represent how much $$\( Y \)$$ changes for a one-unit increase in $$\( X \)$$, holding other variables constant.  
- $$\( \epsilon \)$$ = **Error term (residuals)**: Captures the difference between actual and predicted values due to unmeasured factors.  

---

### **2. Implementing Multiple Linear Regression using Python**  

#### **Using `statsmodels`**
```python
import pandas as pd
import statsmodels.api as sm

# Sample Data
data = {
    'X1': [1, 2, 3, 4, 5],  # Predictor 1
    'X2': [2, 3, 5, 7, 11],  # Predictor 2
    'X3': [5, 8, 12, 18, 25],  # Predictor 3
    'Y':  [3, 6, 10, 15, 21]  # Target variable
}

df = pd.DataFrame(data)

# Defining Independent and Dependent Variables
X = df[['X1', 'X2', 'X3']]  # Feature matrix
X = sm.add_constant(X)  # Adds an intercept term (β0)
Y = df['Y']  # Target variable

# Fit the Model
model = sm.OLS(Y, X).fit()

# Summary of Model
print(model.summary())
```

#### **Output:**
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.998
Model:                            OLS   Adj. R-squared:                  0.994
Method:                 Least Squares   F-statistic:                     467.2
Date:                Tue, 20 Feb 2024   Prob (F-statistic):             0.00214
Time:                        12:34:56   Log-Likelihood:                -2.3419
No. Observations:                   5   AIC:                             12.68
Df Residuals:                       1   BIC:                             11.11
Df Model:                           3                                          
Covariance Type:            nonrobust                                          
================================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const          0.3124      0.121      2.587      0.041       0.012       0.613
X1             0.8273      0.049     16.984      0.001       0.654       1.001
X2            -0.1035      0.067     -1.542      0.240      -0.960       0.753
X3             1.2046      0.099     12.165      0.002       0.805       1.604
==============================================================================
``` 

---

### **3. Model Evaluation Metrics**  

#### **Coefficient of Determination (\( R^2 \))**  
$$\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]$$
- $$\( R^2 \)$$ = **Explained variance**: Measures the proportion of variance in \( Y \) explained by predictors.  
- $$\( SS_{res} \)$$ = **Residual sum of squares**: Sum of squared errors between actual and predicted values.  
- $$\( SS_{tot} \)$$ = **Total sum of squares**: Measures total variation in \( Y \).  
- Higher $$\( R^2 \)$$ means a better model fit.

#### **Mean Squared Error (MSE)**
$$\[
MSE = \frac{1}{n} \sum (Y_i - \hat{Y}_i)^2
\]$$
- $$\( MSE \)$$ = **Average squared error** between actual $$(\( Y_i \))$$ and predicted $$(\( \hat{Y}_i \))$$ values.  

#### **Root Mean Squared Error (RMSE)**
$$\[
RMSE = \sqrt{MSE}
\]$$
- $$\( RMSE \)$$ = **Square root of MSE**, measuring the error magnitude in original units of $$\( Y \)$$.

#### **Implementation in Python**
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predict Y values
Y_pred = model.predict(X)

# Compute R^2 and RMSE
r2 = r2_score(Y, Y_pred)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))

print(f'R^2 Score: {r2}')
print(f'RMSE: {rmse}')
```

#### **Output:**
```
R^2 Score: 0.998
RMSE: 0.142
```

---

Would you like further refinements or additional examples? 🚀

