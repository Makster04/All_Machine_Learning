## Evaluation Metrics used for Regression Models vs Classification Problems

In machine learning, classification and regression are two different types of problems, and the evaluation metrics for each type of problem are designed to assess model performance in different ways. Here's a comparison of the evaluation metrics used for classification problems versus those used for regression models:

### **Classification Metrics**  
Classification problems involve predicting categorical outcomes, where the goal is to assign an input to one of several classes. Common evaluation metrics for classification include:

1. **Accuracy**:
   - The ratio of correctly predicted instances to the total instances in the dataset.
   - Formula: $$\(\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}\)$$

2. **Precision**:
   - The proportion of positive predictions that are actually correct.
   - Formula: $$\(\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}\)$$

3. **Recall (Sensitivity)**:
   - The proportion of actual positives that are correctly identified by the model.
   - Formula: $$\(\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}\)$$

4. **F1-Score**:
   - The harmonic mean of Precision and Recall, used when there is an imbalance between classes.
   - Formula: $$\(\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)$$

5. **Confusion Matrix**:
   - A matrix that summarizes the performance of the classification model, showing True Positives, False Positives, True Negatives, and False Negatives.

6. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
   - Measures the performance of a binary classification model by evaluating its ability to distinguish between classes.
   - The AUC value ranges from 0 to 1, where 1 indicates a perfect classifier.

7. **Log Loss (Cross-Entropy Loss)**:
   - Measures the performance of a classification model by calculating the difference between the predicted probabilities and the actual outcomes.
   - Used for models that output probabilities (e.g., logistic regression).

### **Regression Metrics**  
Regression problems involve predicting continuous outcomes, where the goal is to predict a real-valued output. Common evaluation metrics for regression include:

1. **Mean Absolute Error (MAE)**:
   - The average of the absolute differences between the predicted values and actual values.
   - Formula: $$\(\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|\)$$

2. **Mean Squared Error (MSE)**:
   - The average of the squared differences between the predicted values and actual values.
   - Formula: $$\(\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\)$$

3. **Root Mean Squared Error (RMSE)**:
   - The square root of the MSE, which brings the error measure back to the original unit of the output variable.
   - Formula: $$\(\text{RMSE} = \sqrt{\text{MSE}}\)$$

4. **R-squared (R²)**:
   - Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
   - Formula: $$\(\text{R}^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\)$$
   - R² ranges from 0 to 1, with 1 indicating perfect prediction.

5. **Adjusted R-squared**:
   - A modified version of R² that adjusts for the number of predictors in the model. It is used to penalize models with too many predictors.
   - Formula: $$\(\text{Adjusted R}^2 = 1 - \left( \frac{n-1}{n-p-1} \right) (1 - \text{R}^2)\)$$
   - Where $$\(n\)$$ is the number of data points and \(p\) is the number of predictors.

6. **Mean Absolute Percentage Error (MAPE)**:
   - Measures the accuracy of the model by calculating the percentage difference between the predicted and actual values.
   - Formula: $$\(\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\)$$

7. **Huber Loss**:
   - A combination of MSE and MAE that is less sensitive to outliers than MSE and more sensitive than MAE.
   - Formula: $$\( \text{Huber}(y_i, \hat{y}_i) = \begin{cases} 
\frac{1}{2}(y_i - \hat{y}_i)^2 & \text{for } |y_i - \hat{y}_i| \leq \delta \\
\delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2 & \text{for } |y_i - \hat{y}_i| > \delta
\end{cases} \)$$

### **Key Differences**:
- **Classification** metrics focus on categorical outcomes (e.g., True/False), performance on the positive/negative class, and handling imbalances, whereas **regression** metrics focus on the magnitude of error in continuous outputs.
- **Classification** uses metrics like Accuracy, Precision, Recall, and F1-Score, while **regression** uses metrics like MAE, MSE, RMSE, and R².
- **Regression metrics** typically measure the "distance" between predicted and actual values (absolute or squared), while **classification metrics** assess the model's ability to assign instances correctly to predefined categories.

Each of these metrics helps provide insights into different aspects of a model's performance depending on whether the problem is classification or regression.

---
## Overfitting Indicators for Regression Models vs Classification Models:

### **Overfitting Indicators for Regression Models**

| **Metric/Function**        | **Result of Overfitting**                              | **Explanation**                                                                                                                                                            |
|----------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Mean Absolute Error (MAE)** | Low on training data, high on testing/validation data (e.g., Training MAE = 0.5, Testing MAE = 3.0) | The model performs well on training data but struggles to generalize to unseen data, leading to higher MAE on testing/validation data.                                      |
| **Mean Squared Error (MSE)**  | Low on training data, high on testing/validation data (e.g., Training MSE = 0.25, Testing MSE = 4.5) | Similar to MAE, overfitting is indicated by a large difference in MSE between training and testing data.                                                                  |
| **Root Mean Squared Error (RMSE)** | Low on training data, high on testing/validation data (e.g., Training RMSE = 0.5, Testing RMSE = 2.5) | Overfitting is identified when the RMSE is much lower on training data compared to testing data, showing that the model has memorized the training data.                  |
| **R-squared (R²)**           | High on training data, low on testing/validation data (e.g., Training R² = 0.98, Testing R² = 0.60) | A high R² on training data but a low R² on testing/validation data suggests that the model fits the training data well but doesn't generalize to new data.               |
| **Adjusted R-squared**      | High on training data, low on testing/validation data (e.g., Training Adjusted R² = 0.95, Testing Adjusted R² = 0.55) | This metric adjusts R² based on the number of predictors, so if overfitting occurs, the adjusted R² will drop significantly on the testing/validation data.              |
| **Mean Absolute Percentage Error (MAPE)** | Low on training data, high on testing/validation data (e.g., Training MAPE = 5%, Testing MAPE = 25%) | A low MAPE on training data with a high MAPE on testing/validation indicates that the model is not able to generalize well to unseen data.                                |
| **Huber Loss**              | Low on training data, high on testing/validation data (e.g., Training Huber Loss = 0.1, Testing Huber Loss = 1.2) | Similar to MSE, but more robust to outliers; a significant difference between training and testing performance suggests overfitting.                                       |

### **Overfitting Indicators for Classification Problems**

| **Metric/Function**        | **Result of Overfitting**                              | **Explanation**                                                                                                                                                            |
|----------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Accuracy**               | High on training data, low on testing/validation data (e.g., Training Accuracy = 98%, Testing Accuracy = 75%) | Overfitting is indicated when accuracy is significantly higher on training data than on testing/validation data, showing the model memorizes the training set.             |
| **Precision**              | High on training data, low on testing/validation data (e.g., Training Precision = 0.95, Testing Precision = 0.70) | If precision is high on training but much lower on testing/validation, the model may be overfitting by favoring memorized instances rather than generalizing.               |
| **Recall**                 | High on training data, low on testing/validation data (e.g., Training Recall = 0.93, Testing Recall = 0.60) | A large drop in recall between training and testing data suggests overfitting, as the model memorizes specific cases but fails to generalize.                              |
| **F1-Score**               | High on training data, low on testing/validation data (e.g., Training F1-Score = 0.94, Testing F1-Score = 0.68) | If the F1-score is much higher on training data than on testing data, this points to overfitting, where precision and recall are high on training but suffer on new data.    |
| **Confusion Matrix**       | High number of True Positives on training, low on testing (e.g., Training True Positives = 950, Testing True Positives = 600) | A confusion matrix showing a high number of True Positives on training data but a significant drop on testing data can indicate overfitting.                             |
| **ROC-AUC**                | High on training data, low on testing/validation data (e.g., Training AUC = 0.98, Testing AUC = 0.75) | Overfitting may manifest as a high AUC on training data but a low AUC on testing data, suggesting the model is over-optimizing on the training set.                      |
| **Log Loss (Cross-Entropy Loss)** | Low on training data, high on testing/validation data (e.g., Training Log Loss = 0.1, Testing Log Loss = 1.5) | A significant increase in log loss from training to testing data suggests that the model is overfitting, producing incorrect probabilistic predictions on unseen data.      |
| **Confusion Matrix (for Imbalanced Classes)** | Very low false negatives/positives on training, high on testing (e.g., Training False Positives = 10, Testing False Positives = 50) | In an imbalanced classification problem, overfitting could be indicated by a low number of false negatives and false positives on training data, but much higher on testing data. |

### **Key Observations:**
- **Regression Models**: If a regression model shows a significant difference between the performance on the training data and the testing data (e.g., much lower error on the training set), this suggests overfitting.
- **Classification Problems**: Similarly, if a classification model shows high accuracy, precision, recall, or F1-score on the training set but these metrics drop substantially on the testing data (e.g., large gap in metrics), this is an indication of overfitting.

These metrics help identify overfitting, where the model has learned the training data well but fails to generalize to unseen data.
