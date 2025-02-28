# Evaluation Metrics used for Regression Models vs Classification Problems
### 1. **Algorithms (Regression vs Classification)**  
Here’s the revised table with regression algorithms listed first:  

| Algorithm                          | Regression | Classification |
|-------------------------------------|------------|---------------|
| Linear Regression                  | ✅          | ❌             |
| Ridge & Lasso Regression           | ✅          | ❌             |
| Polynomial Regression              | ✅          | ❌             |
| Decision Tree                      | ✅          | ✅             |
| Random Forest                      | ✅          | ✅             |
| Support Vector Machine (SVM)       | ✅          | ✅             |
| K-Nearest Neighbors (KNN)          | ✅          | ✅             |
| Neural Networks                    | ✅          | ✅             |
| Gradient Boosting (XGBoost, LightGBM) | ✅       | ✅             |
| Logistic Regression                | ❌          | ✅             |
| Naïve Bayes                        | ❌          | ✅             |  

---

### 2. **Metrics Used for Regression vs Classification**  
| Metric | Regression | Classification |
|--------|-----------|---------------|
| **Mean Squared Error (MSE)** | ✅ | ❌ |
| **Root Mean Squared Error (RMSE)** | ✅ | ❌ |
| **Mean Absolute Error (MAE)** | ✅ | ❌ |
| **R-Squared (R²)** | ✅ | ❌ |
| **Adjusted R-Squared** | ✅ | ❌ |
| **Accuracy** | ❌ | ✅ |
| **Precision** | ❌ | ✅ |
| **Recall (Sensitivity)** | ❌ | ✅ |
| **F1-Score** | ❌ | ✅ |
| **ROC-AUC Score** | ❌ | ✅ |
| **Log-Loss** | ❌ | ✅ |

---

### 3. **How to Determine if an Algorithm is Overfitting**  
#### **For Regression Models:**
- **Low training error, but high test error** (High variance)
- **R² is very high (close to 1) on training but much lower on test**
- **MSE or RMSE is significantly lower on training than test**
- **Overly complex model (high-degree polynomial regression)**

### **How to Determine if an Algorithm is Overfitting**  

#### **For Regression Models:**  
- **Low training error, but high test error** (**High variance**)  
  *(Example: Training MSE = 2.3, Test MSE = 15.7)*  
- **R² is very high (close to 1) on training but much lower on test**  
  *(Example: Training R² = 0.98, Test R² = 0.45)*  
- **MSE or RMSE is significantly lower on training than test**  
  *(Example: Training RMSE = 1.5, Test RMSE = 7.8)*  
- **Overly complex model (high-degree polynomial regression)**  
  *(Example: A 10th-degree polynomial fits every training point perfectly but performs poorly on new data.)*  

#### **For Classification Models:**  
- **Training accuracy is much higher than test accuracy**  
  *(Example: Training Accuracy = 98%, Test Accuracy = 65%)*  
- **High precision but very low recall (or vice versa)**  
  *(Example: Precision = 95%, Recall = 40%, indicating the model is too strict and missing positive cases)*  
- **Perfect separation of training data (likely memorized)**  
  *(Example: Decision tree with depth = 20 perfectly classifies all training points but fails on unseen data.)*  
- **ROC-AUC close to 1 on training but significantly lower on test**  
  *(Example: Training AUC = 0.99, Test AUC = 0.60, meaning the model struggles with generalization.)*  

#### **General Indicators for Both:**  
- **Adding more training data does not improve performance**  
  *(Example: Even after adding 10,000 more samples, test accuracy remains at 65% while training is still 98%.)*  
- **Removing features does not significantly affect accuracy**  
  *(Example: Dropping 5 features changes Training Accuracy from 97% to 96%, indicating redundancy.)*  
- **Cross-validation shows large differences between training and validation scores**  
  *(Example: 5-fold CV results: Training Accuracy = 96%, Validation Accuracy = 70%.)*  

---

Here’s your request formatted in the same style:

---

### **Regression Metrics**  
Regression problems involve predicting continuous outcomes, where the goal is to predict a real-valued output. Common evaluation metrics for regression include:

1. **Mean Squared Error (MSE)**  
   - Measures the average squared difference between predicted and actual values.  
   - Larger errors are penalized more heavily due to squaring.  
   - Formula: $$\(\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2\)$$  
   ```python
   from sklearn.metrics import mean_squared_error
   mse = mean_squared_error(y_true, y_pred)
   print(f'MSE: {mse}')
   ```

2. **Root Mean Squared Error (RMSE)**  
   - The square root of MSE, bringing errors back to the original unit.  
   - Formula: $$\(\text{RMSE} = \sqrt{\text{MSE}}\)$$  
   ```python
   from sklearn.metrics import mean_squared_error
   rmse = mean_squared_error(y_true, y_pred, squared=False)
   print(f'RMSE: {rmse}')
   ```

3. **Mean Absolute Error (MAE)**  
   - Measures the average absolute difference between predicted and actual values.  
   - Less sensitive to large errors than MSE.  
   - Formula: $$\(\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|\)$$  
   ```python
   from sklearn.metrics import mean_absolute_error
   mae = mean_absolute_error(y_true, y_pred)
   print(f'MAE: {mae}')
   ```

4. **R-Squared (R²)**  
   - Represents how well the independent variables explain the variance in the dependent variable.  
   - Ranges from 0 to 1, where 1 is a perfect fit.  
   - Formula: $$\(\text{R}^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\)$$  
   ```python
   from sklearn.metrics import r2_score
   r2 = r2_score(y_true, y_pred)
   print(f'R²: {r2}')
   ```

5. **Adjusted R-Squared**  
   - Adjusts R² for the number of predictors, penalizing overly complex models.  
   - Formula: $$\(\text{Adjusted R}^2 = 1 - \left( \frac{n-1}{n-p-1} \right) (1 - \text{R}^2)\)$$  
   ```python
   n = len(y_true)  # Number of observations
   p = X.shape[1]   # Number of predictors
   adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
   print(f'Adjusted R²: {adjusted_r2}')
   ```

---

### **Classification Metrics**  
Classification problems involve predicting categorical outcomes, where the goal is to assign an input to one of several classes. Common evaluation metrics for classification include:

1. **Accuracy**  
   - The proportion of correct predictions.  
   - Formula: $$\(\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}\)$$  
   ```python
   from sklearn.metrics import accuracy_score
   accuracy = accuracy_score(y_true, y_pred)
   print(f'Accuracy: {accuracy}')
   ```

2. **Precision**  
   - Measures how many predicted positives are actually correct.  
   - Formula: $$\(\text{Precision} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}\)$$  
   ```python
   from sklearn.metrics import precision_score
   precision = precision_score(y_true, y_pred, average='binary')
   print(f'Precision: {precision}')
   ```

3. **Recall (Sensitivity)**  
   - Measures how many actual positives were correctly predicted.  
   - Formula: $$\(\text{Recall} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}\)$$  
   ```python
   from sklearn.metrics import recall_score
   recall = recall_score(y_true, y_pred, average='binary')
   print(f'Recall: {recall}')
   ```

4. **F1-Score**  
   - The harmonic mean of precision and recall, useful for imbalanced datasets.  
   - Formula: $$\(\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)$$  
   ```python
   from sklearn.metrics import f1_score
   f1 = f1_score(y_true, y_pred, average='binary')
   print(f'F1-Score: {f1}')
   ```

5. **ROC-AUC Score**  
   - Evaluates the model’s ability to distinguish between classes using probabilities.  
   - A score of 1 means perfect classification.  
   ```python
   from sklearn.metrics import roc_auc_score
   auc = roc_auc_score(y_true, y_pred_prob)
   print(f'ROC-AUC: {auc}')
   ```

6. **Log-Loss**  
   - Measures how far predicted probabilities deviate from actual labels.  
   ```python
   from sklearn.metrics import log_loss
   logloss = log_loss(y_true, y_pred_prob)
   print(f'Log Loss: {logloss}')
   ```

---

Now all metrics are structured the same way! 🚀
### **Key Differences**:
- **Classification** metrics focus on categorical outcomes (e.g., True/False), performance on the positive/negative class, and handling imbalances, whereas **regression** metrics focus on the magnitude of error in continuous outputs.
- **Classification** uses metrics like Accuracy, Precision, Recall, and F1-Score, while **regression** uses metrics like MAE, MSE, RMSE, and R².
- **Regression metrics** typically measure the "distance" between predicted and actual values (absolute or squared), while **classification metrics** assess the model's ability to assign instances correctly to predefined categories.

Each of these metrics helps provide insights into different aspects of a model's performance depending on whether the problem is classification or regression.

---
## Overfitting Indicators for Regression Models vs Classification Models:


| **Metric/Function**        | **Result of Overfitting**                              | **Explanation**                                                                                                                                                            |
|----------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Mean Absolute Error (MAE)** | Low on training data, high on testing/validation data (e.g., Training MAE = 0.5, Testing MAE = 3.0) | The model performs well on training data but struggles to generalize to unseen data, leading to higher MAE on testing/validation data.                                      |
| **Mean Squared Error (MSE)**  | Low on training data, high on testing/validation data (e.g., Training MSE = 0.25, Testing MSE = 4.5) | Similar to MAE, overfitting is indicated by a large difference in MSE between training and testing data.                                                                  |
| **Root Mean Squared Error (RMSE)** | Low on training data, high on testing/validation data (e.g., Training RMSE = 0.5, Testing RMSE = 2.5) | Overfitting is identified when the RMSE is much lower on training data compared to testing data, showing that the model has memorized the training data.                  |
| **R-squared (R²)**           | High on training data, low on testing/validation data (e.g., Training R² = 0.98, Testing R² = 0.60) | A high R² on training data but a low R² on testing/validation data suggests that the model fits the training data well but doesn't generalize to new data.               |
| **Adjusted R-squared**      | High on training data, low on testing/validation data (e.g., Training Adjusted R² = 0.95, Testing Adjusted R² = 0.55) | This metric adjusts R² based on the number of predictors, so if overfitting occurs, the adjusted R² will drop significantly on the testing/validation data.              |
| **Mean Absolute Percentage Error (MAPE)** | Low on training data, high on testing/validation data (e.g., Training MAPE = 5%, Testing MAPE = 25%) | A low MAPE on training data with a high MAPE on testing/validation indicates that the model is not able to generalize well to unseen data.                                |
| **Huber Loss**              | Low on training data, high on testing/validation data (e.g., Training Huber Loss = 0.1, Testing Huber Loss = 1.2) | Similar to MSE, but more robust to outliers; a significant difference between training and testing performance suggests overfitting.                                       |


### **Key Observations:**
- **Regression Models**: If a regression model shows a significant difference between the performance on the training data and the testing data (e.g., much lower error on the training set), this suggests overfitting.
- **Classification Problems**: Similarly, if a classification model shows high accuracy, precision, recall, or F1-score on the training set but these metrics drop substantially on the testing data (e.g., large gap in metrics), this is an indication of overfitting.

These metrics help identify overfitting, where the model has learned the training data well but fails to generalize to unseen data.

---

## Example Use of doing all the metrics:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Example for Classification Problem:
y_true_class = [1, 0, 1, 1, 0, 1, 0]
y_pred_class = [1, 0, 1, 0, 0, 1, 1]
y_pred_prob_class = [0.9, 0.2, 0.95, 0.4, 0.1, 0.8, 0.7]  # Predicted probabilities for ROC-AUC & Log Loss

print(f"Accuracy: {accuracy_score(y_true_class, y_pred_class)}")
print(f"Precision: {precision_score(y_true_class, y_pred_class)}")
print(f"Recall: {recall_score(y_true_class, y_pred_class)}")
print(f"F1-Score: {f1_score(y_true_class, y_pred_class)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_true_class, y_pred_class)}")
print(f"ROC-AUC: {roc_auc_score(y_true_class, y_pred_prob_class)}")
print(f"Log Loss: {log_loss(y_true_class, y_pred_prob_class)}")

# Example for Regression Problem:
y_true_reg = [3.5, 2.8, 4.1, 5.0, 3.3]
y_pred_reg = [3.6, 2.9, 3.9, 4.8, 3.5]

print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg)}")
print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg)}")
print(f"RMSE: {mean_squared_error(y_true_reg, y_pred_reg, squared=False)}")
print(f"R²: {r2_score(y_true_reg, y_pred_reg)}")
print(f"Adjusted R²: {1 - (1 - r2_score(y_true_reg, y_pred_reg)) * (len(y_true_reg) - 1) / (len(y_true_reg) - len(y_pred_reg) - 1)}")
```
---

## Decision Tree falls under either:
Yes, **decision trees** can be used for both **classification** and **regression** tasks, depending on the problem you're trying to solve.

- **Classification Decision Tree**: If the decision tree is used for classifying data into categories (e.g., spam vs. non-spam), it is evaluated using **classification metrics** like **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

- **Regression Decision Tree**: If the decision tree is used for predicting continuous values (e.g., predicting house prices), it is evaluated using **regression metrics** like **mean absolute error (MAE)**, **mean squared error (MSE)**, **root mean squared error (RMSE)**, and **R-squared (R²)**.

### Python Implementation for Decision Tree:

#### **For Classification (Decision Tree Classifier):**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Example data
X_train_class = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train_class = [0, 0, 1, 1]

# Fit model
clf = DecisionTreeClassifier()
clf.fit(X_train_class, y_train_class)

# Make predictions
y_pred_class = clf.predict(X_train_class)

# Evaluate
print(f"Accuracy: {accuracy_score(y_train_class, y_pred_class)}")
print(f"Precision: {precision_score(y_train_class, y_pred_class)}")
print(f"Recall: {recall_score(y_train_class, y_pred_class)}")
print(f"F1-Score: {f1_score(y_train_class, y_pred_class)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_train_class, y_pred_class)}")
```

#### **For Regression (Decision Tree Regressor):**
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Example data
X_train_reg = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train_reg = [1.5, 2.5, 3.5, 4.5]

# Fit model
regressor = DecisionTreeRegressor()
regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = regressor.predict(X_train_reg)

# Evaluate
print(f"MAE: {mean_absolute_error(y_train_reg, y_pred_reg)}")
print(f"MSE: {mean_squared_error(y_train_reg, y_pred_reg)}")
print(f"RMSE: {mean_squared_error(y_train_reg, y_pred_reg, squared=False)}")
print(f"R²: {r2_score(y_train_reg, y_pred_reg)}")
```

### Summary of Decision Trees:
- **Decision Tree Classifier** is for **classification tasks** and uses classification metrics (accuracy, precision, recall, etc.).
- **Decision Tree Regressor** is for **regression tasks** and uses regression metrics (MAE, MSE, RMSE, etc.). 

The implementation for both is very similar, with the primary difference being the use of `DecisionTreeClassifier` for classification tasks and `DecisionTreeRegressor` for regression tasks.
