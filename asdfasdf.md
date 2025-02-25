Only consider the columns specified in `relevant_columns` when building your model. The next step is to create dummy variables from categorical variables. Remember to drop the first level for each categorical column and make sure all the values are of type `float`: 

```python
# Create dummy variables
relevant_columns = ['Pclass', 'Age', 'SibSp', 'Fare', 'Sex', 'Embarked', 'Survived']
df_relevent = df[relevant_columns]

dummy_dataframe = pd.get_dummies(df_relevent, drop_first=True)

dummy_dataframe = dummy_dataframe.astype(float)

print(dummy_dataframe.shape)
```
