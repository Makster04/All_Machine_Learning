# Data Science Definitions by Mathematical Foundation

---

## 🔷 Linear Algebra

### Representation & Structure

- **Dimension:** The number of features or variables in a dataset. More dimensions mean more complexity.
- **Independent Variables (X):** The inputs used to predict the outcome.
- **Dependent Variable (Y):** The outcome being predicted.
- **Coefficients (β):** The values that determine how much each input affects the outcome.
- **Intercept (β₀):** The starting value when all inputs are zero.
- **Slope:** The rate at which one variable changes in response to another.
- **Binary Columns:** Data columns that contain only two values, like 0 and 1.
- **Multicollinearity:** When inputs are too similar, making it hard to tell their effects apart.

### Encoding & Transformation

- **One-Hot Encoding:** Turning categories into separate binary columns (e.g., "Red," "Blue," "Green" into three separate columns).
- **Dummy Variables:** A way to represent categories using numbers (e.g., "Male" = 0, "Female" = 1).
- **StandardScaler:** A tool that adjusts data so different features are on the same scale.
- **MinMaxScaler:** A method that scales data between 0 and 1.
- **Standardized:** Adjusting data so it has a mean of zero and a standard deviation of one.
- **Polynomial:** A transformation that allows models to capture more complex patterns.

### Models & Methods

- **Linear Regression:** A way to predict an outcome using a straight-line relationship between variables.
- **OLS (Ordinary Least Squares):** A method in regression that finds the best-fitting line by minimizing squared errors.
- **Sparse Model:** A model that ignores many features by setting their weights to zero.
- **Ridge:** A method that reduces large weights in regression models to prevent overfitting.
- **Lasso:** A method that removes less important inputs by setting some weights to zero.
- **Pipeline:** An automated process that handles data preparation, training, and testing.

---

## 🔶 Calculus & Optimization

### Core Optimization

- **Gradient Descent:** An optimization algorithm that minimizes a loss function. In linear regression, it minimizes MSE to find the best-fit line. In logistic regression, it minimizes log loss (cross-entropy loss) to improve probability predictions.
- **Loss Function:** A formula that calculates how far predictions are from actual values, helping guide model improvement.
- **Cost Function:** A formula that measures a model's error, helping guide improvements.
- **Error Loss Function:** A function that calculates how far predictions are from actual results.
- **Optimization:** The process of fine-tuning a model's settings to reduce errors and improve performance.

### Regularization

- **Regularization:** A technique to prevent overfitting by adding penalties for complexity.
- **Hyperparameter:** A setting chosen before training, like how fast a model learns.

### Curve Fitting & Models

- **Logistic Regression:** A model that predicts probabilities for binary classification problems.
- **Polynomial Regression:** Similar to linear regression but allows curves instead of straight lines.
- **Recursive Process:** A recursive process is when a function calls itself repeatedly, breaking a problem into smaller parts until it reaches a simple stopping point (base case).

### Model Complexity

- **Bias-Variance Trade-off:** In machine learning, refers to balancing model simplicity (bias) and complexity (variance) to minimize error and improve generalization to unseen data.
- **Overfitting:** Happens when a model remembers the training data too well but struggles to make correct predictions on new data. Too high of variance.
- **Underfitting:** When a model is too simple and fails to find patterns in the data.
- **Generalization:** In machine learning, the model's ability to perform well on unseen data by recognizing underlying patterns, not memorizing.

---

## 🟡 Statistics & Probability

### Core Statistical Concepts

- **Parameter:** A fixed number in a model, like an average or weight in an equation.
- **Variance:** A measure of how spread out data points are. Higher variance means more variability.
- **Continuous Value:** A number that can take any value, like height or temperature.
- **Discrete Value:** A countable number, like the number of apples in a basket.

### Classification & Decision Boundaries

- **Classification:** Task of categorizing data into predefined classes or labels based on input features, such as spam detection or image recognition.
- **Decision Boundary:** The exact boundary a model creates to separate different classes in the data.
- **Boundary:** A line or region that separates different categories in a classification model.
- **Binary Model:** A model that predicts only two possible outcomes, like "yes" or "no."
- **Threshold:** The cutoff point where a model decides between classes based on probability, affecting errors like false positives and false negatives.
- **Decision Tree:** A flowchart-like model that splits data into branches based on feature conditions to make predictions or classifications.

### Classification Metrics

- **Accuracy:** The percentage of correct predictions out of all predictions made.
- **Precision (TP/(TP+FP)):** Measures how many predicted positive cases are actually correct. High precision means fewer false positives.
- **Recall / Sensitivity (TP/(TP+FN)):** Measures how many actual positive cases are correctly identified. High recall is important when missing positives is costly.
- **F1-Score:** The average of precision and recall, balancing both.
- **F-Score:** A weighted harmonic mean of precision and recall. It depends on the model's threshold, structure, and data.
- **Harmonic Mean:** A metric balancing precision and recall. It gives more weight to lower values, preventing extreme bias toward one metric.
- **Specificity (True Negative Rate):** The percentage of true negatives out of all actual negatives.
- **AUC-ROC:** Measures the model's ability to tell apart classes, with higher values being better.
- **Log-Loss (Cross-Entropy Loss):** A measure of how well the model's probability predictions match the actual outcomes, with penalties for more confident incorrect predictions.
- **Confusion Matrix:** A table that shows how well a classification model performed, with counts of true positives, false positives, true negatives, and false negatives.
- **True Positive (TP):** A correctly predicted positive instance where the model detects a real positive case.
- **True Negative (TN):** A correctly predicted negative instance where the model accurately identifies a non-positive case.
- **False Positive (FP):** An incorrect prediction where the model mistakenly classifies a negative instance as positive.
- **False Negative (FN):** An incorrect prediction where the model fails to detect a true positive case.

### Regression Metrics

- **Mean Squared Error (MSE):** Measures the average of squared differences between predicted and actual values. Larger errors are penalized more. Lower values are better.
- **Mean Absolute Error (MAE):** The average of absolute differences between predicted and actual values. It's less affected by large errors than MSE.
- **Root Mean Squared Error (RMSE):** The square root of MSE, showing the error in the same units as the data.
- **R-Squared (R²):** Shows how well the model explains the data. Values close to 1 are good, values close to 0 mean it's not explaining much.
- **Adjusted R-Squared:** Similar to R-squared, but adjusts for the number of predictors in the model, helping to avoid overfitting.
- **Mean Absolute Percentage Error (MAPE):** Measures the average percentage error between predicted and actual values. It's useful when the scale of the data matters.
- **Explained Variance Score:** Shows how much of the data's variation is explained by the model.

### Probability & Prediction

- **Y-Score:** The probability of the positive class. It represents how confident the model is that a given instance belongs to the positive class, typically ranging from 0 to 1.
- **Information Gain:** A measure used in decision trees to determine the effectiveness of a feature in splitting data. Higher values indicate better splits that reduce uncertainty.

### Sampling & Validation

- **Training:** The phase where a machine learns from data by adjusting its internal settings to minimize mistakes.
- **Testing:** Checking how well a trained model performs on new, unseen data.
- **Training Data:** The dataset used to teach a model.
- **Test Data:** A separate dataset used to check a model's performance.
- **Validation Performance:** How well a model works on a separate dataset used for fine-tuning.
- **K-fold Cross Validation:** A method that splits data into multiple parts to test the model multiple times.
- **Cross Validation:** A way to evaluate a model by testing it on different subsets of data.
- **Scoring Argument:** A setting that defines how a model's performance is measured, like accuracy or precision.
- **Data Leakage:** When information from the test data accidentally influences training, leading to misleading results.
- **Supervised Learning:** Training models using labeled data with known output values used for guiding model training, to predict outcomes or classifications.
- **Unsupervised Learning:** Training models using unlabeled data with unknown outputs, to predict outcomes or classifications.

### Feature Selection

- **Filter Methods:** Feature selection techniques that assess individual features' relevance based on statistical tests, independently of machine learning algorithms.
- **Wrapper Methods:** Feature selection techniques that evaluate subsets of features by training and testing models, optimizing feature combinations through iterative processes like forward or backward selection.
- **Embedded Methods:** Feature selection integrated into model training, where algorithms like decision trees or LASSO perform feature selection during the learning process.
- **Information Gain:** A measure used in decision trees to determine the effectiveness of a feature in splitting data. Higher values indicate better splits that reduce uncertainty.
- **Domain Knowledge:** Expertise in a specific area, providing context and understanding to improve problem-solving, decision-making, and model development.
