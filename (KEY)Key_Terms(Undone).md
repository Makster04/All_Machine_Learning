## Important terms & defintions in ML (Still more to be put)

True Negative (2493)
False Negative (3)
True Positive (0)
False Positive (4)

Not everytime accuracy is doing well (How many examples you clarify correctly), doesnt mean the model will always be doing weel (Massive classify
Class Balance will be a big part

Metric:
Precision (TP/(TP+FP))
Recall (TP/(TP+FN)) Total number of positive examples in the data set (When false negatives are a lot worse)
Harmonic Mean (2(Precison+Recall)/2(TP+FP+FN))

Here are simplified explanations of each term:  

### **General Terms**  
- **Log-Loss Function**: A way to measure how well a classification model predicts probabilities. Lower values mean better accuracy.  
- **Training**: The phase where a machine learns from data by adjusting its internal settings to minimize mistakes.  
- **Testing**: Checking how well a trained model performs on new, unseen data.  
- **Binary Cross-Entropy (BCE)**: A method for evaluating classification models by measuring how close predicted probabilities are to actual results.  
- **Optimization**: The process of fine-tuning a model’s settings to reduce errors and improve performance.  
- **Classification**: Sorting data into groups, like detecting spam emails or recognizing faces in images.  
- **Boundary**: A line or region that separates different categories in a classification model.  
- **Decision Boundary**: The exact boundary a model creates to separate different classes in the data.  
- **Binary Model**: A model that predicts only two possible outcomes, like "yes" or "no."  
- **Loss Function**: A formula that calculates how far predictions are from actual values, helping guide model improvement.  

### **Statistical Concepts**  
- **Parameter**: A fixed number in a model, like an average or weight in an equation.  
- **Variance**: A measure of how spread out data points are. Higher variance means more variability.  
- **Dimension**: The number of features or variables in a dataset. More dimensions mean more complexity.  
- **Continuous Value**: A number that can take any value, like height or temperature.  
- **Discrete Value**: A countable number, like the number of apples in a basket.  

### **Regression Analysis**  
- **Regression**: A method to find relationships between variables and predict future values.  
- **Linear Regression**: A way to predict an outcome using a straight-line relationship between variables.  
- **Polynomial Regression**: Similar to linear regression but allows curves instead of straight lines.  
- **Logistic Regression**: A model that predicts probabilities for binary classification problems.  
- **Linear**: A relationship where changes in one variable consistently affect another.  
- **Standardized**: Adjusting data so it has a mean of zero and a standard deviation of one.  
- **Slope**: The rate at which one variable changes in response to another.  
- **Dependent Variable (Y)**: The outcome being predicted.  
- **Independent Variables (X)**: The inputs used to predict the outcome.  
- **Coefficients (β)**: The values that determine how much each input affects the outcome.  
- **Intercept (β₀)**: The starting value when all inputs are zero.  
- **Dummy Variables**: A way to represent categories using numbers (e.g., "Male" = 0, "Female" = 1).  
- **One-Hot Encoding**: Turning categories into separate binary columns (e.g., "Red," "Blue," "Green" into three separate columns).  
- **Binary Columns**: Data columns that contain only two values, like 0 and 1.  
- **StandardScaler**: A tool that adjusts data so different features are on the same scale.  
- **MinMaxScaler**: A method that scales data between 0 and 1.  
- **Overfitting**: When a model memorizes the training data instead of learning patterns, making it bad at new data.  
- **Underfitting**: When a model is too simple and fails to find patterns in the data.  

### **Model Evaluation Metrics**  
- **R-Squared**: Measures how well a model explains the variation in data.  
- **Adjusted R-Squared**: A version of R-Squared that accounts for the number of inputs.  
- **Mean Absolute Error (MAE)**: The average difference between predictions and actual values.  
- **Mean Squared Error (MSE)**: The average squared differences between predictions and actual values.  
- **Root Mean Squared Error (RMSE)**: Similar to MSE but takes the square root, making it easier to interpret.  
- **Validation Performance**: How well a model works on a separate dataset used for fine-tuning.  
- **Model Performance**: The overall effectiveness of a model.  
- **Multicollinearity**: When inputs are too similar, making it hard to tell their effects apart.  
- **Scoring Argument**: A setting that defines how a model’s performance is measured, like accuracy or precision.  

### **Machine Learning Concepts**  
- **Training Data**: The dataset used to teach a model.  
- **Test Data**: A separate dataset used to check a model’s performance.  
- **Generalizes to Unseen Data**: The model can correctly predict outcomes for new data.  
- **Regularization**: A technique to prevent overfitting by adding penalties for complexity.  
- **Ridge**: A method that reduces large weights in regression models to prevent overfitting.  
- **Lasso**: A method that removes less important inputs by setting some weights to zero.  
- **Cost Function**: A formula that measures a model’s error, helping guide improvements.  
- **Error Loss Function**: A function that calculates how far predictions are from actual results.  
- **OLS (Ordinary Least Squares)**: A method in regression that finds the best-fitting line by minimizing squared errors.  
- **Hyperparameter**: A setting chosen before training, like how fast a model learns.  
- **Polynomial**: A transformation that allows models to capture more complex patterns.  
- **Pipeline**: An automated process that handles data preparation, training, and testing.  
- **Data Leakage**: When information from the test data accidentally influences training, leading to misleading results.  
- **Sparse Model**: A model that ignores many features by setting their weights to zero.  
- **K-fold Cross Validation**: A method that splits data into multiple parts to test the model multiple times.  
- **Cross Validation**: A way to evaluate a model by testing it on different subsets of data.  

This should make the definitions easier to grasp! Let me know if you want any further simplifications or examples. 😊


### Feature Selection
**Feature selection** is the process by which you select a subset of features relevant for model construction. Feature selection comes with several benefits, the most obvious being the improvement in performance of a machine learning algorithm. Other benefits include:
- ***Decrease in computational complexity:*** As the number of features is reduced in a model, the easier it will be to compute the parameters of your model. It will also mean a decrease in the amount of data storage required to maintain the features of your model
- ***Understanding your data:*** In the process of feature selection, you will potentially gain more understanding of how features relate to one another

1. **Domain Knowledge**: Expertise in a specific area, providing context and understanding to improve problem-solving, decision-making, and model development by leveraging relevant information, concepts, and experiences from the field.

2. **Filter Methods**: Feature selection techniques that assess individual features’ relevance based on statistical tests, independently of machine learning algorithms, aiming to improve model performance by removing irrelevant or redundant features.

3. **Wrapper Methods**: Feature selection techniques that evaluate subsets of features by training and testing models, optimizing feature combinations through iterative processes like forward or backward selection to maximize predictive accuracy.

4. **Embedded Methods**: Feature selection integrated into model training, where algorithms like decision trees or LASSO perform feature selection during the learning process, balancing performance and model complexity efficiently.
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

13. **Much better test performance than unregularized polynomial**: L2 regularization helps the polynomial model avoid overfitting, leading to more accurate predictions on test data.
14. **High Bias, Low Weight**:  
The decision boundary will be overly simplistic, possibly linear, with poor fitting to the data. It will have low flexibility and underfit, resulting in poor performance.
---

### Speaking of Graphs
- **High Bias, Low Weight**: The decision boundary will be overly simplistic, possibly linear, with poor fitting to the data. It will have low flexibility and underfit, resulting in poor performance. Probabilities would be poor since the model is too simplistic.

- **Low Bias, High Weight**: The decision boundary will be complex, closely fitting the data. However, high weights may cause overfitting, resulting in a sharp boundary that captures noise, leading to high variance. Probabilities could be misleading because the model may overfit.

- **High Bias, High Weight**: The decision boundary might be complex but still not fit the data well. It could be highly curved, but the model's generalization is poor due to high bias. Probabilities would be inconsistent and inaccurate, as the model struggles to find the correct patterns.

- **Low Bias, Low Weight**: The decision boundary will be moderately flexible, capturing data patterns without overfitting noise. The model should generalize well but may lack sufficient complexity for all patterns. Probabilities would be relatively reliable and well-calibrated.
