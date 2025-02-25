Logistic Regression can be implemented using both `scikit-learn` and `statsmodels`. Below are examples using each library to fit a logistic regression model, followed by a comparison of the two approaches.

### 1. **Logistic Regression using `scikit-learn`**:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'feature1': [2, 3, 1, 5, 4, 6],
    'feature2': [1, 6, 2, 7, 8, 9],
    'target': [0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
```

### 2. **Logistic Regression using `statsmodels`**:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'feature1': [2, 3, 1, 5, 4, 6],
    'feature2': [1, 6, 2, 7, 8, 9],
    'target': [0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Add constant (intercept) to the model
X = sm.add_constant(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression model using statsmodels
model = sm.Logit(y_train, X_train)
result = model.fit()

# Print the summary of the model
print(result.summary())

# Make predictions
y_pred_prob = result.predict(X_test)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
```

### Key Differences:

1. **`scikit-learn`**:
   - Focuses on machine learning and model prediction.
   - It is generally used for predictions and performance evaluation.
   - Does not provide statistical details like p-values and confidence intervals by default.
   - Simpler to implement and more suited for ML tasks.

2. **`statsmodels`**:
   - Focuses on statistical modeling and provides detailed summaries of the model, including p-values, coefficients, and confidence intervals.
   - More suitable for statistical analysis where understanding the model's parameters is important.
   - Offers a more traditional approach to regression analysis.

### Output Example:
- **`scikit-learn`**: Outputs accuracy and confusion matrix.
- **`statsmodels`**: Provides detailed regression results, including coefficients, p-values, and other statistical outputs.

Let me know if you'd like further elaboration on any part!
