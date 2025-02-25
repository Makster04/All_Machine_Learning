## Important terms & defintions in ML (Still more to be put)

### **Statistical Concepts**  
- **Variance**: Measures the spread of data points in a dataset by calculating the average squared difference from the mean.  
- **Parameter**: A numerical characteristic that defines a statistical model or distribution, such as a mean or regression coefficient.  
- **Dimension**: The number of features or variables in a dataset or mathematical space, affecting the complexity of a model.
- **Continuous Value:** A numerical value that can take any real number within a range, including decimals. Examples: height, temperature, weight, and time measurements.
- **Discrete Value:** A numerical value that is countable, finite, and cannot be divided meaningfully. Examples: number of students, cars, books, and apples.

### **Regression Analysis**  
- **Regression**: A statistical technique used to model relationships between a dependent variable and one or more independent variables, predicting outcomes and identifying trends in data.
- **Linear:** Refers to a relationship or function where changes in one variable correspond proportionally to changes in another.
- **Linear Regression**: Models relationships between dependent and independent variables using a straight line. Minimizes residuals using least squares to predict continuous values.
- **Polynomial Regression:** Extends linear regression by fitting nonlinear relationships using polynomial terms. Captures curvatures in data, useful for complex patterns.
- **Logistic Regression:** Estimates probabilities for binary classification problems using a sigmoid function, predicting discrete outcomes rather than continuous values.
- **Slope**: The rate of change of the dependent variable per unit increase in the independent variable in a linear equation.  
- **Dependent Variable (Y)**: The outcome variable that the model aims to predict based on input features.  
- **Independent Variables (X)**: The predictor variables used in a model to estimate the dependent variable.  
- **Coefficients (β)**: Weights assigned to independent variables in regression, representing their influence on the dependent variable.  
- **Intercept (β₀)**: The predicted value of the dependent variable when all independent variables are zero.  
- **Dummy Variables**: Binary indicators (0 or 1) representing categorical data for regression or classification models.  
- **One-Hot Encoding**: A transformation method that converts categorical variables into multiple binary columns, each representing a unique category.
- **Binary Columns**: Columns in a dataset that contain only two possible values, typically 0 or 1. These columns represent categorical data in a binary format, often used for classification tasks.
- **StandardScaler:** A preprocessing technique in machine learning that standardizes features by removing the mean and scaling to unit variance. It transforms data to have a mean of 0 and a standard deviation of 1.
- **Cross Validation:** A model evaluation technique where the dataset is split into multiple subsets (folds). The model is trained on some folds and tested on the remaining fold(s), repeating the process to assess performance.
- **K-fold Cross validation:** expands on the idea of training and test splits by splitting the entire dataset into K equal sections of data. 
- **Training data:** Used to train the model. The model learns patterns, relationships, and parameters from this data.
- **Test data:** Used to evaluate the model's performance after training. It helps assess how well the model generalizes to unseen data.
-  **Generalizes to unseen data:** It means the model performs well on new, previously unseen data that wasn't part of its training set.
-  **Scoring Argument:** refers to a parameter used to define how a model's performance is evaluated. It specifies the metric (e.g., accuracy, precision) used during model evaluation.

### **Model Evaluation Metrics**  
- **Adjusted R-Squared**: A refined version of R-Squared that adjusts for the number of predictors, preventing overestimation.  
- **R-Squared**: A statistical measure that explains the proportion of variance in the dependent variable accounted for by independent variables.  
- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values, measuring prediction accuracy.  
- **Root Mean Squared Error (RMSE)**: The square root of the mean squared differences between predicted and actual values, penalizing larger errors.
- **Mean Squared Error (MSE):** measures the average squared difference between predicted and actual values, reflecting the model's accuracy. Lower MSE indicates better model performance, with fewer errors in predictions.
- **Validation Performance**: The accuracy or effectiveness of a model measured on unseen validation data.  
- **Model Performance**: The overall effectiveness of a predictive model, evaluated using metrics like accuracy, MAE, and RMSE.
- **Multicollinearity:** When independent variables in a regression model are highly correlated, making it hard to determine their individual effects and leading to unreliable coefficient estimates.

### **Machine Learning Concepts**  
- **Regularization**: A technique that adds penalties to model complexity to prevent overfitting, such as L1 (Lasso) or L2 (Ridge).  
- **Training Data**: The dataset used to train a machine learning model, allowing it to learn patterns and relationships.  
- **Ridge**: A regression technique that applies L2 regularization, adding a penalty to large coefficients to prevent overfitting. We believe may all have some predictive power. Unlike Lasso, it reduces model complexity without eliminating features.
- **Lasso:** A regularization method for linear regression that applies an L1 penalty, shrinking coefficients to zero and performing feature selection by excluding irrelevant variables, helping to prevent overfitting. Unlike Ridge, Lasso performs feature selection by setting some coefficients to zero, making it better for sparse models.
- **Cost Function**: A mathematical function that quantifies the error of a model’s predictions, guiding optimization.  
- **Error Loss Function**: A function that measures the discrepancy between predicted and actual values, helping to minimize prediction errors.
- **OLS (Ordinary Least Squares):** A statistical method used in linear regression to estimate the relationship between variables by minimizing the sum of squared differences between observed and predicted values.
- **Hyperparameter:** A parameter set before model training that controls the learning process, such as learning rate or number of layers in a neural network. It is not learned from data.
- **Polynomial:** Refer to transformations of input features into higher-degree terms (e.g., squared, cubic) to capture non-linear relationships, enhancing model flexibility for complex patterns in data.
- **Data Leakage**: Leads to overconfident estimates of model performance during the validation and testing phases.
- **Pipeline**: A sequential workflow that automates data preprocessing, feature engineering, model training, and evaluation, ensuring reproducibility, efficiency, and avoiding data leakage by applying transformations consistently.
- **Sparse Model**: A model where many feature coefficients are zero, meaning only a subset of features significantly contribute, improving interpretability and reducing computational complexity.

L1 metric is absolute magnitude of the weights 

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
