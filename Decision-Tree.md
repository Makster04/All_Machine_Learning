New notes being addded:
Gini Impurity: Heavily penalizes heterrogenity (more strongly than entropy)

- Decision Trees can over-split which could lead to over-fitting, too much purity.
- Regulaizer:
- Max depth: DecisionTreeClassifer(max_depth=_)- You will have to run multiple iteriations to fix an overfit and impure decision tree
- min sample split= Minimum number of samles allowed to split- Where I should stop on the minimizing to make sure it doesnt lead to the overfitting.
- How useful is each fature 
- 
Choose a 
- If I say "no", it splits one feature space into one class
- Split the data features X and target Y
- Big advantage of tree-ased, you dont have to scale and no need to transform
- categorical features can be handleed easily
- You can use this instread of finding weights
- Decision Trees are blindandly fast
  
1. Split data features and target
2. Make a decision (a split) based on some notion that given split aids in sepaFrating different classes in feature space.
3. Continue on with each partition
