Scikit-learn provides several ways to perform cross-validation, each catering to different types of datasets and model evaluation needs. Here are the key methods:

### 1. **K-Fold Cross-Validation (`KFold`)**:
   - **Description**: The dataset is split into `k` subsets or folds. The model is trained `k` times, each time using `k-1` folds for training and the remaining fold for testing.
   - **Usage**:
     ```python
     from sklearn.model_selection import KFold, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     kf = KFold(n_splits=10, shuffle=True, random_state=42)
     scores = cross_val_score(model, X, y, cv=kf)
     print("Scores:", scores)
     ```

### 2. **Stratified K-Fold Cross-Validation (`StratifiedKFold`)**:
   - **Description**: Similar to K-Fold, but it ensures that each fold has a proportional representation of the target variable. This is particularly useful for imbalanced datasets.
   - **Usage**:
     ```python
     from sklearn.model_selection import StratifiedKFold, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
     scores = cross_val_score(model, X, y, cv=skf)
     print("Scores:", scores)
     ```

### 3. **Leave-One-Out Cross-Validation (LOOCV) (`LeaveOneOut`)**:
   - **Description**: For each sample in the dataset, the model is trained on all other samples, with one data point left out as the test set. This is repeated for each data point.
   - **Usage**:
     ```python
     from sklearn.model_selection import LeaveOneOut, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     loocv = LeaveOneOut()
     scores = cross_val_score(model, X, y, cv=loocv)
     print("Scores:", scores)
     ```

### 4. **Leave-P-Out Cross-Validation (`LeavePOut`)**:
   - **Description**: A generalization of LOOCV, where `p` samples are left out for testing, and the model is trained on the remaining `n-p` samples.
   - **Usage**:
     ```python
     from sklearn.model_selection import LeavePOut, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     lpo = LeavePOut(p=2)
     scores = cross_val_score(model, X, y, cv=lpo)
     print("Scores:", scores)
     ```

### 5. **ShuffleSplit Cross-Validation (`ShuffleSplit`)**:
   - **Description**: The dataset is randomly shuffled and split into a specified number of train-test sets. Each split is independent of the others.
   - **Usage**:
     ```python
     from sklearn.model_selection import ShuffleSplit, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
     scores = cross_val_score(model, X, y, cv=ss)
     print("Scores:", scores)
     ```

### 6. **Stratified ShuffleSplit Cross-Validation (`StratifiedShuffleSplit`)**:
   - **Description**: Similar to `ShuffleSplit`, but ensures that each train-test split maintains the proportion of each class in the target variable, useful for imbalanced datasets.
   - **Usage**:
     ```python
     from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
     scores = cross_val_score(model, X, y, cv=sss)
     print("Scores:", scores)
     ```

### 7. **TimeSeriesSplit Cross-Validation (`TimeSeriesSplit`)**:
   - **Description**: Used for time series data, where the data is split based on time, ensuring that the training set consists of past data, and the test set consists of future data.
   - **Usage**:
     ```python
     from sklearn.model_selection import TimeSeriesSplit, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris

     data = load_iris()
     X, y = data.data, data.target

     model = RandomForestClassifier()

     tscv = TimeSeriesSplit(n_splits=10)
     scores = cross_val_score(model, X, y, cv=tscv)
     print("Scores:", scores)
     ```

### 8. **Group K-Fold Cross-Validation (`GroupKFold`)**:
   - **Description**: Similar to K-Fold but groups the data based on certain groupings, ensuring that data from the same group are either in the training or the testing set, but not both.
   - **Usage**:
     ```python
     from sklearn.model_selection import GroupKFold, cross_val_score
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.datasets import load_iris
     import numpy as np

     data = load_iris()
     X, y = data.data, data.target
     groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # Example of group labels

     model = RandomForestClassifier()

     gkf = GroupKFold(n_splits=2)
     scores = cross_val_score(model, X, y, groups=groups, cv=gkf)
     print("Scores:", scores)
     ```

### Conclusion:

Scikit-learn provides multiple ways to perform cross-validation, allowing for flexibility depending on your dataset and modeling needs. You can choose between simple methods like K-Fold, or more specialized methods like Stratified K-Fold, TimeSeriesSplit, or Group K-Fold. The choice of cross-validation method depends on factors such as dataset size, class distribution, and whether the data has a temporal or group structure.
