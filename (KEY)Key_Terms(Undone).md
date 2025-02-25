## Important terms & defintions in ML (Still more to be put)

### Other terms or definitions (Yet to be placed in a category):
- **Log-Loss Function:** Log-Loss function (Logarithmic Loss) measures a classification model’s performance by quantifying the difference between predicted probabilities and actual binary labels.
- **Training**: The process where a machine learning model learns patterns from labeled data by adjusting its parameters to minimize errors using optimization algorithms.  
- **Testing**: The evaluation phase where a trained model is assessed on unseen data to measure its performance, generalization ability, and accuracy in real-world scenarios.
- **Binary Cross-Entropy (BCE)** is a loss function measuring the difference between predicted and actual binary labels, used in classification tasks to optimize model accuracy via logarithmic loss.
- **Optimization**: The process of adjusting model parameters to minimize a loss function and improve performance, often using algorithms like Gradient Descent.  
- **Classification**: A supervised learning task where a model categorizes input data into predefined classes, such as spam detection or image recognition.  
- **Boundary**: A dividing line or surface in feature space that separates different categories in a classification model.  
- **Decision Boundary**: The specific boundary learned by a classification algorithm that separates different classes in the dataset, guiding predictions.  
- **Optimization:** No closed form solution for w and b minmizing cost function
- **Binary Model:** A machine learning model that predicts one of two possible outcomes (e.g., 0 or 1) in classification tasks like spam detection or medical diagnosis.
- **Loss Function:** A mathematical function that quantifies the difference between predicted and actual values, guiding model optimization by minimizing errors during training.

### Statistical Concepts
1. **Parameter**: A numerical characteristic that defines a statistical model or distribution, such as a mean or regression coefficient.
2. **Variance**: Measures the spread of data points in a dataset by calculating the average squared difference from the mean.
3. **Dimension**: The number of features or variables in a dataset or mathematical space, affecting the complexity of a model.
4. **Continuous Value**: A numerical value that can take any real number within a range, including decimals. Examples: height, temperature, weight, and time measurements.
5. **Discrete Value**: A numerical value that is countable, finite, and cannot be divided meaningfully. Examples: number of students, cars, books, and apples.

---

### Regression Analysis
1. **Regression**: A statistical technique used to model relationships between a dependent variable and one or more independent variables, predicting outcomes and identifying trends in data.
2. **Linear Regression**: Models relationships between dependent and independent variables using a straight line. Minimizes residuals using least squares to predict continuous values.
3. **Polynomial Regression**: Extends linear regression by fitting nonlinear relationships using polynomial terms. Captures curvatures in data, useful for complex patterns.
4. **Logistic Regression**: Estimates probabilities for binary classification problems using a sigmoid function, predicting discrete outcomes rather than continuous values.
5. **Linear**: Refers to a relationship or function where changes in one variable correspond proportionally to changes in another.
6. **Standardized**: Refers to the process of rescaling data to have a mean of zero and a standard deviation of one, ensuring features are on the same scale.
7. **Slope**: The rate of change of the dependent variable per unit increase in the independent variable in a linear equation.
8. **Dependent Variable (Y)**: The outcome variable that the model aims to predict based on input features.
9. **Independent Variables (X)**: The predictor variables used in a model to estimate the dependent variable.
10. **Coefficients (β)**: Weights assigned to independent variables in regression, representing their influence on the dependent variable.
11. **Intercept (β₀)**: The predicted value of the dependent variable when all independent variables are zero.
12. **Dummy Variables**: Binary indicators (0 or 1) representing categorical data for regression or classification models.
13. **One-Hot Encoding**: A transformation method that converts categorical variables into multiple binary columns, each representing a unique category.
14. **Binary Columns**: Columns in a dataset that contain only two possible values, typically 0 or 1. These columns represent categorical data in a binary format, often used for classification tasks.
15. **StandardScaler**: A preprocessing technique in machine learning that standardizes features by removing the mean and scaling to unit variance. It transforms data to have a mean of 0 and a standard deviation of 1.
16. **MinMaxScaler**: A feature scaling technique that transforms data into a specified range, typically [0, 1], by subtracting the minimum value and dividing by the range.
17. **Overfitting**: A model becomes too complex, capturing noise and irrelevant patterns, leading to high training accuracy but poor generalization on new data.
18. **Underfitting**: A model is too simple, failing to capture underlying patterns, resulting in poor performance on both training and unseen data.

---

### Model Evaluation Metrics
1. **R-Squared**: A statistical measure that explains the proportion of variance in the dependent variable accounted for by independent variables.
2. **Adjusted R-Squared**: A refined version of R-Squared that adjusts for the number of predictors, preventing overestimation.
3. **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values, measuring prediction accuracy.
4. **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values, reflecting the model's accuracy. Lower MSE indicates better model performance, with fewer errors in predictions.
5. **Root Mean Squared Error (RMSE)**: The square root of the mean squared differences between predicted and actual values, penalizing larger errors.
6. **Validation Performance**: The accuracy or effectiveness of a model measured on unseen validation data.
7. **Model Performance**: The overall effectiveness of a predictive model, evaluated using metrics like accuracy, MAE, and RMSE.
8. **Multicollinearity**: When independent variables in a regression model are highly correlated, making it hard to determine their individual effects and leading to unreliable coefficient estimates.
9. **Scoring Argument**: Refers to a parameter used to define how a model's performance is evaluated. It specifies the metric (e.g., accuracy, precision) used during model evaluation.

---

### Machine Learning Concepts
1. **Training Data**: The dataset used to train a machine learning model, allowing it to learn patterns and relationships.
2. **Test Data**: Used to evaluate the model's performance after training. It helps assess how well the model generalizes to unseen data.
3. **Generalizes to Unseen Data**: It means the model performs well on new, previously unseen data that wasn't part of its training set.
4. **Regularization**: A technique that adds penalties to model complexity to prevent overfitting, such as L1 (Lasso) or L2 (Ridge).
5. **Ridge**: A regression technique that applies L2 regularization, adding a penalty to large coefficients to prevent overfitting. It reduces model complexity without eliminating features.
6. **Lasso**: A regularization method for linear regression that applies an L1 penalty, shrinking coefficients to zero and performing feature selection by excluding irrelevant variables, helping to prevent overfitting.
7. **Cost Function**: A mathematical function that quantifies the error of a model’s predictions, guiding optimization.
8. **Error Loss Function**: A function that measures the discrepancy between predicted and actual values, helping to minimize prediction errors.
9. **OLS (Ordinary Least Squares)**: A statistical method used in linear regression to estimate the relationship between variables by minimizing the sum of squared differences between observed and predicted values.
10. **Hyperparameter**: A parameter set before model training that controls the learning process, such as learning rate or number of layers in a neural network.
11. **Polynomial**: Refers to transformations of input features into higher-degree terms (e.g., squared, cubic) to capture non-linear relationships, enhancing model flexibility for complex patterns in data.
12. **Pipeline**: A sequential workflow that automates data preprocessing, feature engineering, model training, and evaluation, ensuring reproducibility, efficiency, and avoiding data leakage by applying transformations consistently.
13. **Data Leakage**: Leads to overconfident estimates of model performance during the validation and testing phases.
14. **Sparse Model**: A model where many feature coefficients are zero, meaning only a subset of features significantly contribute, improving interpretability and reducing computational complexity.
15. **K-fold Cross Validation**: Expands on the idea of training and test splits by splitting the entire dataset into K equal sections of data.
16. **Cross Validation**: A model evaluation technique where the dataset is split into multiple subsets (folds). The model is trained on some folds and tested on the remaining fold(s), repeating the process to assess performance.


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
