# OOP with Sckit-Learn Summary

Scikit-learn provides a robust framework for machine learning in Python. The library is structured around four main classes that help streamline the workflow for building models. Below are explanations, definitions, examples, and Python code snippets related to each class:
1. Estimator
2. Transformer
3. Prdictor
4. Model

#### Mutable vs Non-Mutable
- **Mutable types:** Can be changed after creation. Examples include list, dict, set, numpy.ndarray, and pandas.DataFrame.
- **Immutable types:** Cannot be changed after creation. Examples include int, float, tuple, str, and frozenset.


---

### 1. **Estimator**
**Definition**:  
An **Estimator** is any object that can learn from data. It implements a `fit()` method that takes training data and extracts patterns. All machine learning models, transformers, and predictors in Scikit-learn are derived from the `BaseEstimator` class.

|             | Estimator |
|-------------|-----------|
| **StandardScaler** | ✅ |
| **PCA** | ✅  |
| **KMeans** | ✅ |
| **LinearRegression** | ✅ |

**Example**:  
A `LinearRegression` model is an example of an estimator.

**Python Code**:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Creating some sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Creating an estimator (Linear Regression model)
estimator = LinearRegression()

# Fitting the model to the data
estimator.fit(X, y)

print("Coefficients:", estimator.coef_)
print("Intercept:", estimator.intercept_)
```
Output:
```
Coefficients: [2.]
Intercept: 0.0
```
**Result**:
The model fits a perfect line with equation y = 2X. This result is good, as the model captures the relationship perfectly.
---

### 2. **Transformer**
**Definition**:  
A **Transformer** is a type of estimator that transforms input data in some way. It implements two key methods:
- `fit()`: Learns patterns from data (if necessary).
- `transform()`: Applies the learned transformation to new data.

Some transformers also support `fit_transform()`, which combines both steps.

|                      | Estimator | Transformer | 
|----------------------|-----------|-------------|
| **StandardScaler**   | ✅ | ✅ | 
| **PCA**              | ✅ | ✅ |
| **KMeans**           | ✅ | ✅ |
| **LinearRegression** | ✅ | ❌ |


**Example**:  
`StandardScaler` is a transformer that standardizes features by removing the mean and scaling to unit variance.

**Python Code**:
```python
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Creating a transformer
scaler = StandardScaler()

# Fitting and transforming the data
X_scaled = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Transformed Data:\n", X_scaled)
```
Output:
```
Original Data:
 [[1 2]
  [3 4]
  [5 6]]
Transformed Data:
 [[-1.22474487 -1.22474487]
  [ 0.          0.        ]
  [ 1.22474487  1.22474487]]
```
**Result**:
The data is transformed to have mean 0 and standard deviation 1. This is good, as scaling often improves model performance.

---

### 3. **Predictor**
**Definition**:  
A **Predictor** is any estimator that can make predictions using a `predict()` method. All classifiers and regressors in Scikit-learn are predictors.

|                      | Estimator | Transformer | Predictor |
|----------------------|-----------|-------------|-----------|
| **StandardScaler**   | ✅ | ✅ | ❌ | 
| **PCA**              | ✅ | ✅ | ❌ |
| **KMeans**           | ✅ | ✅ | ✅ |
| **LinearRegression** | ✅ | ❌ | ✅ |

**Example**:  
A `DecisionTreeClassifier` is an example of a predictor.

**Python Code**:
```python
from sklearn.tree import DecisionTreeClassifier

# Sample data
X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
y = np.array([0, 1, 1, 0])

# Creating and fitting a predictor
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# Making a prediction
prediction = classifier.predict([[1.5, 1.5]])
print("Prediction:", prediction)
```
Output:
```
Prediction = 1.0
```
**Result:**
The model classifies the input [1.5, 1.5] as 1. If it generalizes well on unseen data, this is good.

#### Note:
Some additional examples of predictors are:
- Linear 
- `LogisticRegression`: a classifier that uses the logistic regression algorithm
- `KNeighborsRegressor`: a regressor that uses the k-nearest neighbors algorithm
- `LinearReggresion`: a regressor modeling relationships using linear equations.
---

### 4. **Model**
**Definition**:  
A **Model** is a more general term encompassing both estimators and predictors. It refers to a fitted instance of an algorithm that can make predictions on new data. A model is typically an instance of a predictor that has been trained.

|                      | Estimator | Transformer | Predictor | Model |
|----------------------|-----------|-------------|-----------|-------|
| **StandardScaler**   | ✅ | ✅ | ❌ | ❌
| **PCA**              | ✅ | ✅ | ❌ | ✅
| **KMeans**           | ✅ | ✅ | ✅ | ✅
| **LinearRegression** | ✅ | ❌ | ✅ | ✅

**Example**:  
A trained `RandomForestClassifier` is a model, then you can use the useful `Score` method, which typically returns the accuracy for classification problems.

**Python Code**:
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
y = np.array([0, 1, 1, 0])

# Creating a model
model = RandomForestClassifier(n_estimators=10)

# Training the model
model.fit(X, y)

# Evaluating the model using score method (accuracy)
accuracy = model.score(X, y)

print("Model Accuracy:", accuracy)

# Making a prediction
prediction = model.predict([[1.5, 1.5]])
print("Model Prediction:", prediction)
```
Output:
```
Model Accuracy: 1.0
Model Prediction: [1]
```
**Result:**
The model achieves perfect accuracy (1.0), which is excellent on this small dataset, but may not generalize well to larger, unseen data.
---

### Summary Table:
| Class       | Purpose |
|-------------|---------|
| **Estimator** | If it has a `fit()` method, it's an estimator |
| **Transformer** | If it has a `transform()` method, it's a transformer |
| **Predictor** | If it has a `predict()` method, it's a predictor|
| **Model** | If it has a `score()` method, it's a model |

These classes form the foundation of Scikit-learn and are used throughout its machine learning workflow.

### Overlapping Classes

As stated previously, these scikit-learn classes are not mutually exclusive.
- ```StandardScaler``` is an estimator and a transformer BUT NOT a predictor or a model.

- ```LinearRegression``` is an estimator, a predictor, and a model BUT NOT a transformer.

- ```KMeans``` is an estimator, a transformer, a predictor, and a model.

- ```PCA``` is an estimator, a transformer, and a model BUT NOT a predictor.

---

### Confused about these classes?

**Standard Scaler**
1. A Standard Scaler is a technique used to standardize or scale your features (input data) to make them have zero mean and unit variance.
2. The idea is to transform your data so that each feature has a similar scale and the model can perform better, especially for algorithms sensitive to feature scaling (like linear models, SVM, or KNN).

**PCA (Principal Component Analysis)**
1. Dimensionality reduction technique used to reduce the number of variables in a dataset while retaining as much of the original variability (or information) as possible.
2. This is often used when dealing with high-dimensional data to make it more manageable, easier to visualize, and often helps with improving performance in machine learning tasks.

**K-Means (K-Means Clustering)**
1. An unsupervised machine learning algorithm used for clustering *(cluster refers to a group of data points that are similar to each other based on some measure of similarity or distance.)*
2. The goal of K-Means is to divide a set of data points into 𝐾 clusters based on their similarity.

**Linear Reggression**
1. One of the most basic and widely used supervised machine learning algorithms.
2. It models the relationship between a dependent variable *(also called the target variable, 𝑦)* and one or more independent variables *(also called features or predictors, 𝑋)* by fitting a linear equation to the observed data.

---

### Other Notes
#### What is the difference between Linear and Logistic?
- **Linear Regression:** Use When the target variable is continuous (e.g., predicting prices, heights, temperatures).
- **Example:** Predicting the price of a house based on its features (size, location, number of rooms).
- **Logistic Regression:** When the target variable is categorical (especially binary classification, e.g., 0 or 1).
- **Example:** Classifying emails as spam (1) or not spam (0).

