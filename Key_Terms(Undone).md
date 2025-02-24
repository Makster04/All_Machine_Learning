## Important terms & defintions in ML (Still more to be put)

### **Statistical Concepts**  
- **Variance**: Measures the spread of data points in a dataset by calculating the average squared difference from the mean.  
- **Parameter**: A numerical characteristic that defines a statistical model or distribution, such as a mean or regression coefficient.  
- **Dimension**: The number of features or variables in a dataset or mathematical space, affecting the complexity of a model.  

### **Regression Analysis**  
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

### **Model Evaluation Metrics**  
- **Adjusted R-Squared**: A refined version of R-Squared that adjusts for the number of predictors, preventing overestimation.  
- **R-Squared**: A statistical measure that explains the proportion of variance in the dependent variable accounted for by independent variables.  
- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values, measuring prediction accuracy.  
- **Root Mean Squared Error (RMSE)**: The square root of the mean squared differences between predicted and actual values, penalizing larger errors.
- **Mean Squared Error (MSE):** measures the average squared difference between predicted and actual values, reflecting the model's accuracy. Lower MSE indicates better model performance, with fewer errors in predictions.
- **Validation Performance**: The accuracy or effectiveness of a model measured on unseen validation data.  
- **Model Performance**: The overall effectiveness of a predictive model, evaluated using metrics like accuracy, MAE, and RMSE.  

### **Machine Learning Concepts**  
- **Regularization**: A technique that adds penalties to model complexity to prevent overfitting, such as L1 (Lasso) or L2 (Ridge).  
- **Training Data**: The dataset used to train a machine learning model, allowing it to learn patterns and relationships.  
- **Ridge**: A regression technique that applies L2 regularization, adding a penalty to large coefficients to prevent overfitting. We believe may all have some predictive power. Want to heavily penalize wight
- **Lasso:** A regularization method for linear regression that applies an L1 penalty, shrinking coefficients to zero and performing feature selection by excluding irrelevant variables, helping to prevent overfitting. Lasso usually has higher weights. Beleive many are not actually 
- **Cost Function**: A mathematical function that quantifies the error of a model’s predictions, guiding optimization.  
- **Error Loss Function**: A function that measures the discrepancy between predicted and actual values, helping to minimize prediction errors.
- **OLS (Ordinary Least Squares):** A statistical method used in linear regression to estimate the relationship between variables by minimizing the sum of squared differences between observed and predicted values.
- **Hyperparameter:** A parameter set before model training that controls the learning process, such as learning rate or number of layers in a neural network. It is not learned from data.
- **Polynomial:** Refer to transformations of input features into higher-degree terms (e.g., squared, cubic) to capture non-linear relationships, enhancing model flexibility for complex patterns in data.

L1 metric is absolute magnitude of the weights 

### Questions & Comments heard(Undone)

- When a Graph is too stiff
- More data leads to less overfitting
- cross-validation: Gives us a way to test statistical robustness of model performacne

- How do we lower the variance?
- Way to limit/deal with high variance?
- Why would you want to standardize/normalize features?

- True test/hold-out test:

- Unregulazied polynomial function

- penalzing makes them more rigid, doesnt make a model that is a true value of the wights)


- Take you best model and put it in the X-train_processed



- Polynomial Model was poor prediciton because error is overfitting 
- L2 Regulazied the Polynomial
- Much better test performance than unregulazed polynominal
