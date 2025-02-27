New notes being addded:
Gini Impurity: Heavily penalizes heterrogenity (more strongly than entropy)

- Decision Trees can over-split which could lead to over-fitting, too much purity.
- 
Choose a 
- If I say "no", it splits one feature space into one class
- Split the data features X and target Y
- Big advantage of tree-ased, you dont have to scale and no need to transform
- categorical features can be handleed easily
- You can use this instread of finding weights
- Decision Trees are blindandly fast
  
1. Split data features and target
2. Make a decision (a split) based on some notion that given split aids in separating different classes in feature space.
3. Continue on with each partition


## Important terms & defintions in ML (Still more to be put)
---

Here’s the categorized list with the definitions included:  

---

### **Feature Engineering & Data Processing**  

- **Feature Scaling & Encoding:**  
  - **Standardized:** Adjusting data so it has a mean of zero and a standard deviation of one.  
  - **StandardScaler:** A tool that adjusts data so different features are on the same scale.  
  - **MinMaxScaler:** A method that scales data between 0 and 1.  
  - **Dummy Variables:** A way to represent categories using numbers (e.g., "Male" = 0, "Female" = 1).  
  - **One-Hot Encoding:** Turning categories into separate binary columns (e.g., "Red," "Blue," "Green" into three separate columns).  
  - **Binary Columns:** Data columns that contain only two values, like 0 and 1.  

- **Statistical Concepts & Data Properties:**  
  - **Parameter:** A fixed number in a model, like an average or weight in an equation.  
  - **Variance:** A measure of how spread out data points are. Higher variance means more variability.  
  - **Dimension:** The number of features or variables in a dataset. More dimensions mean more complexity.  
  - **Continuous Value:** A number that can take any value, like height or temperature.  
  - **Discrete Value:** A countable number, like the number of apples in a basket.  
  - **Multicollinearity:** When inputs are too similar, making it hard to tell their effects apart.  
  - **Bias-variance trade-off:** In machine learning, refers to balancing model simplicity (bias) and complexity (variance) to minimize error and improve generalization to unseen data.  
  - **Generalization:** In machine learning, the model's ability to perform well on unseen data by recognizing underlying patterns, not memorizing.  

---

### **Classification Problems**  

- **Classification & Decision Boundaries:**  
  - **Classification:** Task of categorizing data into predefined classes or labels based on input features, such as spam detection or image recognition.  
  - **Boundary:** A line or region that separates different categories in a classification model.  
  - **Decision Boundary:** The exact boundary a model creates to separate different classes in the data.  
  - **Binary Model:** A model that predicts only two possible outcomes, like "yes" or "no."  
  - **Threshold:** The cutoff point where a model decides between classes based on probability, affecting errors like false positives and false negatives.
  - **Decision Tree:** A flowchart-like model that splits data into branches based on feature conditions to make predictions or classifications.
  - **Information Gain** (After Decision Tree): A measure used in decision trees to determine the effectiveness of a feature in splitting data. Higher values indicate better splits that reduce uncertainty.  

- **Model Evaluation Metrics (For Classification):**  
  - **Log-Loss Function:** A way to measure how well a classification model predicts probabilities. Lower values mean better accuracy.  
  - **Binary Cross-Entropy (BCE):** A method for evaluating classification models by measuring how close predicted probabilities are to actual results.  

---

### **Regression & Statistical Modeling**  

- **Regression Types:**  
  - **Regression:** A method to find relationships between variables and predict future values.  
  - **Linear Regression:** A way to predict an outcome using a straight-line relationship between variables.  
  - **Polynomial Regression:** Similar to linear regression but allows curves instead of straight lines.  
  - **Logistic Regression:** A model that predicts probabilities for binary classification problems.  

- **Regression Methods & Concepts:**  
  - **OLS (Ordinary Least Squares):** A method in regression that finds the best-fitting line by minimizing squared errors.  
  - **Linear:** A relationship where changes in one variable consistently affect another.  
  - **Slope:** The rate at which one variable changes in response to another.  
  - **Intercept (β₀):** The starting value when all inputs are zero.  
  - **Coefficients (β):** The values that determine how much each input affects the outcome.  
  - **Dependent Variable (Y):** The outcome being predicted.  
  - **Independent Variables (X):** The inputs used to predict the outcome.  
  - **Polynomial:** A transformation that allows models to capture more complex patterns.  
  - **Sparse Model:** A model that ignores many features by setting their weights to zero.  

- **Model Evaluation Metrics (For Regression):**  
  - **R-Squared:** Measures how well a model explains the variation in data.  
  - **Adjusted R-Squared:** A version of R-Squared that accounts for the number of inputs.  
  - **Mean Absolute Error (MAE):** The average difference between predictions and actual values.  
  - **Mean Squared Error (MSE):** The average squared differences between predictions and actual values.  
  - **Root Mean Squared Error (RMSE):** Similar to MSE but takes the square root, making it easier to interpret.  

---

### **Functions & Optimization**  

- **Loss Functions & Optimization:**  
  - **Optimization:** The process of fine-tuning a model’s settings to reduce errors and improve performance.  
  - **Loss Function:** A formula that calculates how far predictions are from actual values, helping guide model improvement.  
  - **Cost Function:** A formula that measures a model’s error, helping guide improvements.  
  - **Error Loss Function:** A function that calculates how far predictions are from actual results.  
  - **Gradient Descent:** An optimization algorithm that minimizes a loss function. In linear regression, it minimizes MSE to find the best-fit line. In logistic regression, it minimizes log loss (cross-entropy loss) to improve probability predictions.  

- **Recursive Processes & Mathematical Functions:**  
  - **Recursive Process:** A recursive process is when a function calls itself repeatedly, breaking a problem into smaller parts until it reaches a simple stopping point (base case).  

---

### **Model Training & Evaluation**  

- **Training & Validation:**  
  - **Training:** The phase where a machine learns from data by adjusting its internal settings to minimize mistakes, seen data. 
  - **Testing:** Checking how well a trained model performs on new, unseen data.  
  - **Training Data:** The dataset used to teach a model.  
  - **Test Data:** A separate dataset used to check a model’s performance.  
  - **Validation Performance:** How well a model works on a separate dataset used for fine-tuning.  
  - **Model Performance:** The overall effectiveness of a model.  
  - **Generalizes to Unseen Data:** The model can correctly predict outcomes for new data.  
  - **Overfitting:** When a model memorizes the training data instead of learning patterns, making it bad at new data.  
  - **Underfitting:** When a model is too simple and fails to find patterns in the data.  

- **Regularization & Hyperparameters:**  
  - **Regularization:** A technique to prevent overfitting by adding penalties for complexity.  
  - **Ridge:** A method that reduces large weights in regression models to prevent overfitting.  
  - **Lasso:** A method that removes less important inputs by setting some weights to zero.  
  - **Hyperparameter:** A setting chosen before training, like how fast a model learns.  

- **Cross Validation & Scoring:**  
  - **K-fold Cross Validation:** A method that splits data into multiple parts to test the model multiple times.  
  - **Cross Validation:** A way to evaluate a model by testing it on different subsets of data.  
  - **Scoring Argument:** A setting that defines how a model’s performance is measured, like accuracy or precision.  

---

### **Machine Learning Workflow & Pipelines**  

- **Pipelines & Data Flow:**  
  - **Pipeline:** An automated process that handles data preparation, training, and testing.  
  - **Data Leakage:** When information from the test data accidentally influences training, leading to misleading results.  

---

### Feature Selection
**Feature selection** is the process by which you select a subset of features relevant for model construction. Feature selection comes with several benefits, the most obvious being the improvement in performance of a machine learning algorithm. Other benefits include:
- ***Decrease in computational complexity:*** As the number of features is reduced in a model, the easier it will be to compute the parameters of your model. It will also mean a decrease in the amount of data storage required to maintain the features of your model
- ***Understanding your data:*** In the process of feature selection, you will potentially gain more understanding of how features relate to one another

1. **Domain Knowledge**: Expertise in a specific area, providing context and understanding to improve problem-solving, decision-making, and model development by leveraging relevant information, concepts, and experiences from the field.

2. **Filter Methods**: Feature selection techniques that assess individual features’ relevance based on statistical tests, independently of machine learning algorithms, aiming to improve model performance by removing irrelevant or redundant features.

3. **Wrapper Methods**: Feature selection techniques that evaluate subsets of features by training and testing models, optimizing feature combinations through iterative processes like forward or backward selection to maximize predictive accuracy.

4. **Embedded Methods**: Feature selection integrated into model training, where algorithms like decision trees or LASSO perform feature selection during the learning process, balancing performance and model complexity efficiently.
---

### Evaluation Method
- **True Positive (TP):** A correctly predicted positive instance where the model detects a real positive case.
  
- **True Negative (TN):** A correctly predicted negative instance where the model accurately identifies a non-positive case.
  
- **False Positive (FP):** An incorrect prediction where the model mistakenly classifies a negative instance as positive.
  
- **False Negative (FN):** An incorrect prediction where the model fails to detect a true positive case.
  
- **Precision (TP/(TP+FP)):** Measures how many predicted positive cases are actually correct. High precision means fewer false positives, making it crucial in applications like fraud detection.
  
- **Recall (TP/(TP+FN)):** Measures how many actual positive cases are correctly identified. High recall is important when missing positives is costly, like in medical diagnosis.
  
- **Harmonic Mean (2(Precision + Recall)/2(TP+FP+FN)):** A metric balancing precision and recall. It gives more weight to lower values, preventing extreme bias toward one metric over another in classification tasks.
  
- **F-Score:** A weighted harmonic mean of precision and recall. It depends on the model’s threshold, structure, and data, ensuring balanced evaluation based on application needs.

1. **High Recall, Low Precision**  
   - The model correctly identifies most positive cases (high recall) but also misclassifies many negative cases as positive (low precision).  
   - This is common in applications where missing a true positive is costly, such as medical diagnosis (e.g., detecting cancer) or fraud detection. The model catches most actual cases but has many false alarms.

2. **Low Recall, High Precision**  
   - The model is very strict in labeling something as positive, meaning that when it does, it's usually correct (high precision). However, it misses many true positives (low recall).  
   - This is useful when false positives are costly, such as in spam detection—only very certain emails are classified as spam, but many spam emails might be missed.

3. **High Recall, High Precision**  
   - The ideal scenario: the model correctly identifies most positive cases (high recall) and does so with few false positives (high precision).  
   - This indicates a well-performing model that effectively balances both metrics, though achieving this in practice is challenging.

4. **Low Recall, Low Precision**  
   - The worst-case scenario: the model misses many true positives (low recall) and also makes many incorrect positive predictions (low precision).  
   - This suggests a poorly trained model, possibly due to poor feature selection, insufficient training data, or improper model tuning.
---
#### When we fit Linear regression
- When we fit a line with Linear Regression, we optimise the **intercept** + **slope** x Weight
- When we use Logistic Regression, we optimize a squiggle (S shaped curve)
- When we use t-SNE, we optimize clusters


---

### Questions & Comments heard(Undone)
1. **When a Graph is too stiff**: It lacks flexibility, making it difficult to capture trends or respond to data changes.

2. **More data leads to less overfitting**: Larger datasets allow models to generalize better, reducing the risk of overfitting to noise.

3. **Cross-validation**: It helps assess the model's robustness by testing it on different data splits, ensuring consistent performance.

4. **How do we lower the variance?**: By using regularization, increasing training data, or applying simpler models to reduce overfitting.

5. **Way to limit/deal with high variance?**: Regularization techniques like L1 or L2 penalization and reducing model complexity can help control high variance.

6. **Why would you want to standardize/normalize features?**: To ensure features with different scales contribute equally, improving model performance and stability.

7. **True test/hold-out test**: This test evaluates model performance on unseen data, simulating real-world predictions and checking generalization ability.

8. **Unregularized polynomial function**: Polynomial models without regularization tend to overfit, capturing noise rather than underlying patterns in the data.

9. **Penalizing makes them more rigid, doesn’t make a model that is a true value of the weights**: Penalizing reduces model complexity but doesn't necessarily reflect the true relationship in the data.

10. **Take your best model and put it in the X-train_processed**: After model optimization, place it in the processed training set to evaluate and train with clean data.

11. **Polynomial Model was poor prediction because error is overfitting**: High-degree polynomials can model noise in the training set, leading to poor generalization to new data.

12. **L2 Regularized the Polynomial**: Regularizing the polynomial reduces overfitting, leading to improved model performance and better generalization on unseen data.

14. **Much better test performance than unregularized polynomial**: L2 regularization helps the polynomial model avoid overfitting, leading to more accurate predictions on test data.
    
15. **Y-Score is the probability of the Positive Class**: It represents how confident the model is that a given instance belongs to the positive class, typically ranging from 0 to 1.

16. **Not every time accuracy is doing well (How many examples you classify correctly), doesn't mean the model will always be doing well (Massive misclassification)**: Accuracy can be misleading in imbalanced datasets, as a model may predict the majority class most of the time while failing on critical minority cases.  

17. **Class balance will be a big part**: If classes are imbalanced, metrics like accuracy can be deceptive, requiring alternative evaluation methods such as precision, recall, or F1-score.  

18. **Threshold, there are far more probable but less equal**: Adjusting the decision threshold can change the balance between precision and recall, affecting the trade-off between false positives and false negatives.  

19. **Metric**: Choosing the right metric (e.g., precision, recall, F1-score, AUC-ROC) depends on the problem context, especially when dealing with imbalanced classes or varying misclassification costs.
20. **When you use Gradient Descent**: if you use it to optimise a straight line, then we will have learned the strategy that optimizes a squiggle and clusters
---

### Speaking of Graphs

- **C** typically controls the trade-off between bias and variance in models. A high C value makes the model more complex (lower bias, higher variance), while a low C value makes it simpler (higher bias, lower variance).
- **High Bias, Low Weight**: The decision boundary will be overly simplistic, possibly linear, with poor fitting to the data. It will have low flexibility and underfit, resulting in poor performance. Probabilities would be poor since the model is too simplistic.

- **Low Bias, High Weight**: The decision boundary will be complex, closely fitting the data. However, high weights may cause overfitting, resulting in a sharp boundary that captures noise, leading to high variance. Probabilities could be misleading because the model may overfit.

- **High Bias, High Weight**: The decision boundary might be complex but still not fit the data well. It could be highly curved, but the model's generalization is poor due to high bias. Probabilities would be inconsistent and inaccurate, as the model struggles to find the correct patterns.

- **Low Bias, Low Weight**: The decision boundary will be moderately flexible, capturing data patterns without overfitting noise. The model should generalize well but may lack sufficient complexity for all patterns. Probabilities would be relatively reliable and well-calibrated.
