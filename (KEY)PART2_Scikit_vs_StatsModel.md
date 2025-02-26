# Logistic Regression in scikit-learn - Lab

## Introduction 

In this lab, you are going to fit a logistic regression model to a dataset concerning heart disease. Whether or not a patient has heart disease is indicated in the column labeled `'target'`. 1 is for positive for heart disease while 0 indicates no heart disease.

## Objectives

In this lab you will: 

- Fit a logistic regression model using scikit-learn 


## Let's get started!

Run the following cells that import the necessary functions and import the dataset: 


```python
# Import necessary functions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
```


```python
# Import data
df = pd.read_csv('heart.csv')
df.head(5)
```
Output
```
|   | age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|---|-----|-----|----|----------|------|-----|---------|--------|-------|---------|-------|----|------|--------|
| 0 | 63  | 1   | 3  | 145      | 233  | 1   | 0       | 150    | 0     | 2.3     | 0     | 0  | 1    | 1      |
| 1 | 37  | 1   | 2  | 130      | 250  | 0   | 1       | 187    | 0     | 3.5     | 0     | 0  | 2    | 1      |
| 2 | 41  | 0   | 1  | 130      | 204  | 0   | 0       | 172    | 0     | 1.4     | 2     | 0  | 2    | 1      |
| 3 | 56  | 1   | 1  | 120      | 236  | 0   | 1       | 178    | 0     | 0.8     | 2     | 0  | 2    | 1      |
| 4 | 57  | 0   | 0  | 120      | 354  | 0   | 1       | 163    | 1     | 0.6     | 2     | 0  | 2    | 1      |

```

## Define appropriate `X` and `y` 

Recall the dataset contains information about whether or not a patient has heart disease and is indicated in the column labeled `'target'`. With that, define appropriate `X` (predictors) and `y` (target) in order to model whether or not a patient has heart disease.


```python
# Split the data into target and predictors
y = df["target"]
X = df.drop("target", axis=1)
```

## Train- test split 

- Split the data into training and test sets 
- Assign 25% to the test set 
- Set the `random_state` to 0 

N.B. To avoid possible data leakage, it is best to split the data first, and then normalize.


```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

## Normalize the data 

Normalize the data (`X`) prior to fitting the model. 


```python
# Your code here
X =  df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X.head(5)
```
Output:
```
|    | age      | sex  | cp       | trestbps | chol     | fbs  | restecg | thalach  | exang | oldpeak  | slope | ca   | thal     | target |
|----|---------|------|---------|----------|---------|------|---------|---------|------|---------|------|------|---------|--------|
| 0  | 0.708333 | 1.0  | 1.000000 | 0.481132 | 0.244292 | 1.0  | 0.0     | 0.603053 | 0.0  | 0.370968 | 0.0  | 0.0  | 0.333333 | 1.0    |
| 1  | 0.166667 | 1.0  | 0.666667 | 0.339623 | 0.283105 | 0.0  | 0.5     | 0.885496 | 0.0  | 0.564516 | 0.0  | 0.0  | 0.666667 | 1.0    |
| 2  | 0.250000 | 0.0  | 0.333333 | 0.339623 | 0.178082 | 0.0  | 0.0     | 0.770992 | 0.0  | 0.225806 | 1.0  | 0.0  | 0.666667 | 1.0    |
| 3  | 0.562500 | 1.0  | 0.333333 | 0.245283 | 0.251142 | 0.0  | 0.5     | 0.816794 | 0.0  | 0.129032 | 1.0  | 0.0  | 0.666667 | 1.0    |
| 4  | 0.583333 | 0.0  | 0.000000 | 0.245283 | 0.520548 | 0.0  | 0.5     | 0.702290 | 1.0  | 0.096774 | 1.0  | 0.0  | 0.666667 | 1.0    |
```

## Fit a model

- Instantiate `LogisticRegression`
  - Make sure you don't include the intercept  
  - set `C` to a very large number such as `1e12` 
  - Use the `'liblinear'` solver 
- Fit the model to the training data 


```python
# Instantiate the model
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')

# Fit the model
model_log = logreg.fit(X_train, y_train)
model_log
```
Output:
```
LogisticRegression(C=1000000000000.0, fit_intercept=False, solver='liblinear')
```

## Predict
Generate predictions for the training and test sets. 


```python
# Generate predictions
y_hat_train = logreg.predict(X_train)
y_hat_test = logreg.predict(X_test)

print(y_hat_train)
print(X_test)
```
Output:
```
[0 1 1 0 0 0 0 1 0 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0
 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 0 1 0 0 0 1 0 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 0 0 1 1 0 0
 1 1 1 0 1 0 1 0 1 1 0 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 1 1 1 0 1 0 1 0 0
 0 1 1 0 1 0 1 1 1 0 1 1 0 1 0 1 1 1 1 1 0 0 1 1 0 0 0 0 1 1 1 1 0 1 1 0 1
 1 0 1 1 0 0 1 1 1 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 1 0 0 1 0 1 0
 0 0 1 1 1]
     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \
225   70    1   0       145   174    0        1      125      1      2.6   
152   64    1   3       170   227    0        0      155      0      0.6   
228   59    1   3       170   288    0        0      159      0      0.2   
201   60    1   0       125   258    0        0      141      1      2.8   
52    62    1   2       130   231    0        1      146      0      1.8   
..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   
46    44    1   2       140   235    0        0      180      0      0.0   
160   56    1   1       120   240    0        1      169      0      0.0   
232   55    1   0       160   289    0        0      145      1      0.8   
181   65    0   0       150   225    0        0      114      0      1.0   
27    51    1   2       110   175    0        1      123      0      0.6   

     slope  ca  thal  
225      0   0     3  
152      1   0     3  
228      1   0     3  
201      1   1     3  
...
181      1   3     3  
27       2   0     2  

[76 rows x 13 columns]
```

## How many times was the classifier correct on the training set?


```python
train_residuals = np.abs(y_train - y_hat_train)
print(train_residuals.value_counts(normalize=True))
```
Output:
```
0    0.854626
1    0.145374
Name: target, dtype: float64
```

## How many times was the classifier correct on the test set?


```python
train_residuals = np.abs(y_test - y_hat_test)
print(train_residuals.value_counts(normalize=True))
```
Output:
```
0    0.828947
1    0.171053
Name: target, dtype: float64
```

## Analysis
Describe how well you think this initial model is performing based on the training and test performance. Within your description, make note of how you evaluated performance as compared to your previous work with regression.

```
Well based on the results, its pretty decent. Both 80-86% accurate```
```

## Summary

In this lab, you practiced a standard data science pipeline: importing data, split it into training and test sets, and fit a logistic regression model. In the upcoming labs and lessons, you'll continue to investigate how to analyze and tune these models for various scenarios.

