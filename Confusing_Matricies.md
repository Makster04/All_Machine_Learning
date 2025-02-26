True Positive (TP): A correctly predicted positive instance where the model detects a real positive case.

True Negative (TN): A correctly predicted negative instance where the model accurately identifies a non-positive case.

False Positive (FP): An incorrect prediction where the model mistakenly classifies a negative instance as positive.

False Negative (FN): An incorrect prediction where the model fails to detect a true positive case.

Use sklearn to create confusion matrices

```python
from sklearn.metrics import confusion_matrix
example_labels = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1]
example_preds  = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1]

cf = confusion_matrix(example_labels, example_preds)
cf
```
Output:
```
array([[2, 3],
       [2, 4]])
```

