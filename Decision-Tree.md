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

---

## **Gini Impurity** in Decision Trees

**Gini Impurity** is a metric used to measure the "impurity" or "purity" of a node in a decision tree for classification tasks. It is used to determine how often a randomly chosen element from the dataset would be incorrectly classified if it was randomly labeled according to the distribution of labels in the node.

- **Gini Impurity** takes values between 0 and 1:
  - A **Gini Impurity of 0** means the node is pure (i.e., all instances in the node belong to a single class).
  - A **Gini Impurity of 1** means the node is maximally impure (i.e., the classes are perfectly mixed).

The goal during the construction of a decision tree is to **split** the dataset into subsets that have **low Gini Impurity**, meaning the samples in each subset are more homogenous (more likely to belong to a single class).

### **Gini Impurity Formula:**
For a given node, the Gini Impurity is calculated using the following formula:

$$\[
Gini = 1 - \sum_{i=1}^{C} p_i^2
\]$$

Where:
- \(C\) is the number of classes.
- \(p_i\) is the probability (or proportion) of class \(i\) in the node.

### **Steps to calculate Gini Impurity**:
1. For each class, calculate the proportion of instances of that class in the node.
2. Square the proportion of each class.
3. Sum the squared proportions.
4. Subtract the sum from 1 to get the Gini Impurity.

### **Example**:
Suppose we have a dataset in a node with two classes (Class A and Class B) and the following class distribution:

- 4 instances of Class A
- 6 instances of Class B
- Total instances in the node: 10

The proportions for each class are:
- \(p_A = \frac{4}{10} = 0.4\)
- \(p_B = \frac{6}{10} = 0.6\)

Now, calculate the Gini Impurity:
$$\[
Gini = 1 - (0.4^2 + 0.6^2)
\]$$
$$\[
Gini = 1 - (0.16 + 0.36)
\]$$
$$\[
Gini = 1 - 0.52 = 0.48
\]$$

So, the Gini Impurity of this node is **0.48**.

### **How Gini Impurity is used in Decision Trees**:
1. **Splitting**: When building a decision tree, the algorithm chooses the feature and split that **minimizes** the Gini Impurity of the resulting child nodes. In other words, the algorithm tries to find splits that result in nodes with lower Gini Impurity, meaning the nodes are more "pure."
   
2. **Best Split Selection**: 
   - For every potential split (based on a feature), the Gini Impurity is calculated for the resulting subsets (child nodes).
   - The split that **minimizes** the weighted average of the Gini Impurity for the child nodes is selected as the best split.
   
3. **Stopping Criteria**: If a node has a Gini Impurity of **0**, this means the node is "pure" (i.e., all samples belong to the same class), and no further splitting is necessary.

### **Advantages of Gini Impurity**:
- **Efficiency**: It's computationally efficient because it only requires the class probabilities of the subsets.
- **Interpretability**: The value of Gini Impurity is easy to interpret (0 means pure, 1 means maximally impure).
- **No Need for Prior Probabilities**: Unlike some other metrics (like entropy), Gini does not require the computation of logarithms, making it faster to compute.

### **Summary**:
- **Gini Impurity** is a measure of how often a randomly chosen element from the node would be incorrectly classified.
- It is used in **decision trees** to decide where to split the data.
- A **low Gini Impurity** indicates that the node contains samples from mostly one class (high purity), while a **high Gini Impurity** indicates the samples are mixed from many classes (low purity).
