# Summary on Multiple Linear Regression:

---

**Multiple Linear Regression:**

**Definition**  
Multiple linear regression is an extension of simple linear regression that models the relationship between a dependent variable and two or more independent variables. The goal is to predict the dependent variable by estimating the coefficients for the independent variables. This model can include both continuous and categorical predictors.

**Key Terms:**

- **Dependent Variable (Y):** The outcome variable that the model aims to predict.
- **Independent Variables (X):** The input variables used to predict the dependent variable.
- **Coefficients (β):** The estimated values representing the relationship between each independent variable and the dependent variable.
- **Intercept (β₀):** The value of the dependent variable when all independent variables are equal to zero.
- **Dummy Variables:** Categorical variables transformed into binary variables (0 or 1) for use in regression models.
- **One-Hot Encoding:** A technique to convert categorical variables into binary variables, where each category is represented by a separate column.
- **Adjusted R-Squared:** A modified version of R-Squared that adjusts for the number of predictors, providing a more accurate measure when there are multiple predictors.
- **R-Squared:** The proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Absolute Error (MAE):** The average of the absolute errors between the predicted and actual values.
- **Root Mean Squared Error (RMSE):** The square root of the average of the squared differences between predicted and actual values.

**Equation:**

The equation for a multiple linear regression model is:

\[ Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ϵ \]

Where:
- \( Y \) is the dependent variable.
- \( β₀ \) is the intercept.
- \( β₁, β₂, ..., βₚ \) are the coefficients for each independent variable \( X₁, X₂, ..., Xₚ \).
- \( ϵ \) is the error term (residuals).

**Example:**
Suppose you're predicting the price of a house using square footage, the number of bedrooms, and location. In this case, the dependent variable \( Y \) is the price of the house, and the independent variables are square footage, number of bedrooms, and location.

The model equation might look like this:

$$\[ \text{Price} = β₀ + β₁ \times \text{Square Footage} + β₂ \times \text{Bedrooms} + β₃ \times \text{Location} + ϵ \]$$

Where:
- $$\( β₀ \)$$ is the intercept.
- $$\( β₁, β₂, β₃ \)$$ are the coefficients for square footage, number of bedrooms, and location.
- "Location" would be a categorical variable (e.g., "Urban" or "Suburban"), which would need to be transformed into a dummy variable using one-hot encoding.

**Preprocessing Categorical Variables:**

To include categorical variables in a regression model, they must first be transformed into numerical values using techniques like **One-Hot Encoding**.

For example, a categorical variable "Location" with two categories: "Urban" and "Suburban" would be encoded into two binary columns:
- **Urban:** 1 if the house is located in an urban area, 0 otherwise.
- **Suburban:** 1 if the house is located in a suburban area, 0 otherwise.

To avoid the **dummy variable trap**, which occurs if all categories are included as dummy variables, we drop one of the columns. For example, we might drop the "Suburban" column and use "Urban" as the reference category.

---

### **Code Example (Python using StatsModels):**

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

---

### **Code Example (Python using scikit-learn):**

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

---

### **Evaluation Metrics:**

1. **Adjusted R-Squared:**  
   A more reliable measure for multiple regression models as it accounts for the number of predictors. Higher values indicate a better fit, but it penalizes unnecessary predictors.

   $$\[
   \text{Adjusted } R^2 = 1 - \left(\frac{(1 - R^2)(n - 1)}{n - p - 1}\right)
   \]$$
   Where:
   - $$\( R^2 \)$$ is the coefficient of determination.
   - $$\( n \)$$ is the number of observations.
   - $$\( p \)$$ is the number of predictors.

2. **Mean Absolute Error (MAE):**  
   The average of the absolute errors between predicted and actual values. A lower MAE indicates better model performance.

   $$\[
   MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
   \]$$

3. **Root Mean Squared Error (RMSE):**  
   The square root of the average of squared differences between predicted and actual values. It is sensitive to large errors.

   $$\[
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}
   \]$$

---

### **Key Takeaways:**
- **Multiple Linear Regression** extends simple linear regression to include multiple predictors, both continuous and categorical.
- It helps **control for confounding** variables by considering multiple variables, making it a more reliable predictive tool.
- **Categorical variables** must be encoded into numerical values before use in regression models, with one-hot encoding being the most common technique. Always drop one category to avoid the dummy variable trap.

--- 

This summary combines the definition, terms, example, and evaluation metrics for multiple linear regression, as well as the code examples and explanations for both StatsModels and scikit-learn implementations. It also emphasizes the key takeaways, ensuring a clear understanding of the concept.

