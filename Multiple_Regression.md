### Multiple Linear Regression

#### **Definition**
Multiple linear regression is an extension of simple linear regression that models the relationship between a dependent variable and two or more independent variables. The goal is to predict the dependent variable by estimating the coefficients for the independent variables. This model includes both continuous and categorical predictors.

#### **Terms**
- **Dependent Variable**: The outcome variable that the model aims to predict.
- **Independent Variables**: The input variables used to predict the dependent variable.
- **Coefficients**: The estimated values that represent the relationship between each independent variable and the dependent variable.
- **Intercept**: The value of the dependent variable when all independent variables are equal to zero.
- **Dummy Variables**: Categorical variables that are transformed into binary variables (0 or 1) for use in regression models.
- **One-Hot Encoding**: A technique to convert categorical variables into binary variables where each category is represented by a separate column.
- **Adjusted R-Squared**: A modified version of R-Squared that adjusts for the number of predictors in the model, providing a more accurate measure when there are multiple predictors.
- **R-Squared**: The proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Absolute Error (MAE)**: The average of the absolute errors between the predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of the average of the squared differences between predicted and actual values.

#### **Equation**
The equation for a multiple linear regression model is:

$$\[
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon
\]$$

Where:
- $$\( Y \)$$ is the dependent variable.
- $$\( \beta_0 \)$$ is the intercept.
- $$\( \beta_1, \beta_2, ..., \beta_p \)$$ are the coefficients for each independent variable $$\( X_1, X_2, ..., X_p \)$$.
- $$\( \epsilon \)$$ is the error term (residuals).

#### **Example**
Suppose you're predicting the price of a house using the square footage, number of bedrooms, and location. In this case, the dependent variable $$\( Y \)$$ is the price of the house, and the independent variables are square footage, number of bedrooms, and location.

$$\[
\text{Price} = \beta_0 + \beta_1 \times \text{Square Footage} + \beta_2 \times \text{Bedrooms} + \beta_3 \times \text{Location} + \epsilon
\]$$

Where:
- $$\( \beta_0 \)$$ is the intercept.
- $$\( \beta_1, \beta_2, \beta_3 \)$$ are the coefficients for square footage, number of bedrooms, and location.
- Location would be a categorical variable (e.g., "Urban" or "Suburban"), which would need to be transformed into a dummy variable using one-hot encoding.

#### **Preprocessing Categorical Variables**
To include categorical variables in a regression model, you need to transform them into numerical values using techniques like **One-Hot Encoding**.

For example, consider a categorical variable "Location" with two categories: "Urban" and "Suburban". After one-hot encoding, you would create two binary columns:
- **Urban**: 1 if the house is located in an urban area, 0 otherwise.
- **Suburban**: 1 if the house is located in a suburban area, 0 otherwise.

However, we avoid the **dummy variable trap**, which occurs if all categories are included as dummy variables. To prevent this, you should drop one column, making it the **reference category**. In this case, you might drop the **Suburban** column and use **Urban** as the reference category.

#### **Code Example using Python (StatsModels)**

```python
import pandas as pd
import statsmodels.api as sm

# Example dataset
data = {
    'SquareFootage': [1200, 1500, 1800, 2000],
    'Bedrooms': [2, 3, 3, 4],
    'Location': ['Urban', 'Suburban', 'Urban', 'Suburban'],
    'Price': [300000, 350000, 400000, 450000]
}

df = pd.DataFrame(data)

# One-hot encoding categorical variable 'Location'
df_encoded = pd.get_dummies(df, drop_first=True)

# Independent variables (X)
X = df_encoded[['SquareFootage', 'Bedrooms', 'Location_Urban']]

# Add a constant (intercept) to the independent variables
X = sm.add_constant(X)

# Dependent variable (y)
y = df['Price']

# Build the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())
```

#### **Code Example using Python (scikit-learn)**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Example dataset
data = {
    'SquareFootage': [1200, 1500, 1800, 2000],
    'Bedrooms': [2, 3, 3, 4],
    'Location': ['Urban', 'Suburban', 'Urban', 'Suburban'],
    'Price': [300000, 350000, 400000, 450000]
}

df = pd.DataFrame(data)

# One-hot encode the 'Location' column
encoder = OneHotEncoder(drop='first')
location_encoded = encoder.fit_transform(df[['Location']]).toarray()

# Add the encoded columns back to the dataframe
location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['Location']))
df = pd.concat([df, location_df], axis=1)

# Prepare the features and target
X = df[['SquareFootage', 'Bedrooms', 'Location_Urban']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
```

#### **Evaluation Metrics**
- **Adjusted R-Squared**: More reliable for multiple regression because it accounts for the number of predictors in the model. Higher values indicate better model fit, but it penalizes unnecessary predictors.
  
 $$\[
  \text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2) (n - 1)}{n - p - 1} \right)
  \]$$
  Where:
  - $$\( R^2 \)$$ is the coefficient of determination.
  - $$\( n \)$$ is the number of observations.
  - $$\( p \)$$ is the number of predictors.

- **Mean Absolute Error (MAE)**: The average of the absolute differences between the predicted and actual values. A lower value indicates better performance.

  $$\[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]$$
  
- **Root Mean Squared Error (RMSE)**: The square root of the average of squared differences between the predicted and actual values. It is sensitive to large errors.

  $$\[
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  \]$$

#### **Key Takeaways**
- **Multiple Linear Regression**: Builds on simple linear regression by considering multiple predictors, including both continuous and categorical variables.
- **Control for Confounding**: By including multiple variables in a model, it helps control for confounding effects, making it a more reliable tool for prediction.
- **Categorical Variables**: Must be encoded before use in regression models, with one-hot encoding being the most common method. Always drop one category to avoid the dummy variable trap.


