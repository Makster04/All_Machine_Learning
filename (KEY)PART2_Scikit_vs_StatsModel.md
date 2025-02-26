
# Fitting a Logistic Regression Model - Lab

## Introduction

In the last lesson you were given a broad overview of logistic regression. This included an introduction to two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with `statsmodels`. For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the [Titanic](https://www.kaggle.com/c/titanic/data) shipwreck or not (yes, it's a bit morbid).


## Objectives

In this lab you will: 

* Implement logistic regression with `statsmodels` 
* Interpret the statistical results associated with model parameters

## Import the data

Import the data stored in the file `'titanic.csv'` and print the first five rows of the DataFrame to check its contents. 


```python
import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

df

```

## Define independent and target variables

Your target variable is in the column `'Survived'`. A `0` indicates that the passenger didn't survive the shipwreck. Print the total number of people who didn't survive the shipwreck. How many people survived?


```python
# Total number of people who survived/didn't survive
Total_Survived= df['Survived'].value_counts()
Total_Survived
```
Output:
```
0    549
1    342
Name: Survived, dtype: int64
```

Only consider the columns specified in `relevant_columns` when building your model. The next step is to create dummy variables from categorical variables. Remember to drop the first level for each categorical column and make sure all the values are of type `float`: 


```python
# Create dummy variables
relevant_columns = ['Pclass', 'Age', 'SibSp', 'Fare', 'Sex', 'Embarked', 'Survived']
df_relevent = df[relevant_columns]

dummy_dataframe = pd.get_dummies(df_relevent, drop_first=True)

dummy_dataframe = dummy_dataframe.astype(float)

print(dummy_dataframe.shape)
```
Output:
```
(891, 8)
```

Did you notice above that the DataFrame contains missing values? To keep things simple, simply delete all rows with missing values. 

> NOTE: You can use the [`.dropna()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) method to do this. 


```python
# Drop missing rows
dummy_dataframe = dummy_dataframe.dropna()
dummy_dataframe.shape
```
Output:
```
(714, 8)
```

Finally, assign the independent variables to `X` and the target variable to `y`: 


```python
# Split the data into X and y
y = dummy_dataframe['Survived']
X = dummy_dataframe.drop(columns=['Survived'])
```

## Fit the model

Now with everything in place, you can build a logistic regression model using `statsmodels` (make sure you create an intercept term as we showed in the previous lesson).  

> Warning: Did you receive an error of the form "LinAlgError: Singular matrix"? This means that `statsmodels` was unable to fit the model due to certain linear algebra computational problems. Specifically, the matrix was not invertible due to not being full rank. In other words, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.


```python
# Build a logistic regression model using statsmodels
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = sm.Logit(y_train, X_train)
result = model.fit()

print(result.summary())
```
Output:
```
Optimization terminated successfully.
         Current function value: 0.427312
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               Survived   No. Observations:                  499
Model:                          Logit   Df Residuals:                      491
Method:                           MLE   Df Model:                            7
Date:                Tue, 25 Feb 2025   Pseudo R-squ.:                  0.3661
Time:                        16:45:37   Log-Likelihood:                -213.23
converged:                       True   LL-Null:                       -336.39
Covariance Type:            nonrobust   LLR p-value:                 1.676e-49
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          6.5020      0.834      7.800      0.000       4.868       8.136
Pclass        -1.4192      0.216     -6.555      0.000      -1.843      -0.995
Age           -0.0484      0.010     -4.845      0.000      -0.068      -0.029
SibSp         -0.3306      0.162     -2.044      0.041      -0.648      -0.014
Fare          -0.0033      0.004     -0.892      0.373      -0.011       0.004
Sex_male      -2.7855      0.267    -10.417      0.000      -3.310      -2.261
Embarked_Q    -0.3045      0.719     -0.424      0.672      -1.714       1.105
Embarked_S    -0.5005      0.329     -1.523      0.128      -1.144       0.143
==============================================================================
```

## Analyze results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.

Summary
```
- Highly significant variables: Pclass, Age, Sex_male (p-value < 0.05)
- Not significant variables: Fare, Embarked_Q, and Embarked_S (p-value > 0.05)
- Marginally significant: SibSp (p-value = 0.041)

The model suggests that factors like class, age, and gender have a notable impact on survival, while others like fare and embarkation port do not contribute significantly.

```

## Summary 

Well done! In this lab, you practiced using `statsmodels` to build a logistic regression model. You then interpreted the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Scikit-learn!

---

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
Well based on the results, its pretty decent. Both 80-86% accurate
```

## Summary

In this lab, you practiced a standard data science pipeline: importing data, split it into training and test sets, and fit a logistic regression model. In the upcoming labs and lessons, you'll continue to investigate how to analyze and tune these models for various scenarios.

