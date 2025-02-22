### **Transformations in Data Processing (Linear, Log, etc.)**  

Transformations are used in **data preprocessing** to improve **model performance**, ensure **normal distribution**, or meet **assumptions of statistical models**. The most common transformations include **linear and logarithmic (log) transformations**.  

---

## **1. Linear Transformation**  
🔹 **What is it?**  
A **linear transformation** is a function that maps input data to a new scale **without changing the relative structure**. It follows the equation:  

\[
y = a x + b
\]

🔹 **Why use it?**  
- **Normalization**: Brings values into a fixed range (e.g., 0 to 1).  
- **Standardization**: Centers data around mean 0, variance 1.  
- **Feature scaling** for machine learning.  

### **Example: Min-Max Scaling (Normalization)**
\[
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
\]
- Scales values between **0 and 1**.  

#### **Python Code: Min-Max Scaling**
```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array([[10], [20], [30], [40], [50]])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)
```
**Output:**  
```
[[0.  ]
 [0.25]
 [0.5 ]
 [0.75]
 [1.  ]]
```
✅ **Data is now between 0 and 1.**  

---

### **Example: Standardization (Z-Score Normalization)**
\[
z = \frac{x - \mu}{\sigma}
\]
- **Centers** the data at mean 0 and standard deviation 1.  

#### **Python Code: Standardization**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

print(standardized_data)
```
✅ **Useful for machine learning algorithms sensitive to scale (e.g., SVM, kNN, PCA).**  

---

## **2. Logarithmic Transformation (Log Transform)**
🔹 **What is it?**  
A log transformation **compresses large values** while expanding small values, making skewed data more **normal**.  

🔹 **Formula:**
\[
y = \log(x)
\]

🔹 **Why use it?**  
- **Reduces right skewness** in data.  
- **Stabilizes variance** for statistical models.  
- **Handles exponential growth** (e.g., financial data, population growth).  

### **Example: Applying Log Transform**
```python
import numpy as np

data = np.array([1, 10, 100, 1000, 10000])
log_data = np.log(data)

print(log_data)
```
**Output:**  
```
[0.         2.30258509 4.60517019 6.90775528 9.21034037]
```
✅ **Large values are compressed, reducing skewness.**  

---

## **3. When to Use Log Transformation vs. Linear Transformation?**
| Transformation | When to Use |
|---------------|------------|
| **Linear (Min-Max, Standardization)** | When the data is normally distributed or close to it. |
| **Log Transformation** | When data is **right-skewed** or follows an **exponential growth pattern**. |

---

## **4. Log Transform with Machine Learning**
Many **machine learning models** perform better with **log-transformed data**, especially **linear regression**.

### **Example: Applying Log Transform to Regression Features**
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Creating a dataset
df = pd.DataFrame({
    'income': [2000, 3000, 5000, 7000, 12000],
    'spending': [1000, 1500, 2000, 2500, 4000]
})

# Applying log transform to income
df['log_income'] = np.log(df['income'])

# Train a linear regression model
X = df[['log_income']]
y = df['spending']
model = LinearRegression()
model.fit(X, y)

print("Model Coefficient:", model.coef_[0])
```
✅ **Log-transformed income allows for better model interpretation.**  

---

## **5. When Log Transform is NOT Recommended**
🚨 **Do not use log transformation when:**  
- **Data contains zeros or negatives** (log is undefined for non-positive numbers).  
- **The relationship is already linear** (log transformation is unnecessary).  
- **Data is normally distributed** (it won’t improve anything).  

#### **Handling Zeros in Log Transformation**
Since **log(0) is undefined**, use **log(x + 1)** (Laplace Smoothing):
```python
df['log_income'] = np.log(df['income'] + 1)
```

---

## **6. Comparing Transformations Visually**
Let's visualize how transformations affect data distribution.

```python
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.exponential(scale=2, size=1000)  # Skewed data

plt.figure(figsize=(12, 5))

# Original Data
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True, bins=30)
plt.title("Original Data")

# Log Transformed Data
plt.subplot(1, 2, 2)
sns.histplot(np.log(data), kde=True, bins=30)
plt.title("Log Transformed Data")

plt.show()
```
✅ **The log transformation makes the distribution more normal.**  

---

## **7. Transformations in Scikit-Learn Pipelines**
If using **Scikit-Learn**, you can apply transformations in a pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Define a log transformer
log_transformer = FunctionTransformer(np.log1p, validate=True)

# Create a pipeline with the transformer and a model
pipeline = Pipeline([
    ('log_transform', log_transformer),
    ('model', LinearRegression())
])

# Fit the pipeline
pipeline.fit(X, y)
```
✅ **Ensures log transformation is consistently applied during training and prediction.**  

---

## **Key Takeaways**
| Transformation | Purpose |
|---------------|------------|
| **Min-Max Scaling** | Scales between **0 and 1** (for neural networks, SVMs, kNN). |
| **Standardization (Z-Score)** | Centers data to **mean 0, variance 1** (for PCA, regression). |
| **Log Transformation** | Reduces **right skewness**, stabilizes variance (for exponential growth). |
| **Log(x + 1)** | Used when data contains zeros. |
| **Pipelines** | Automate transformations in ML workflows. |

---

## Non-Linear Tranformation
## **Non-Linear Transformations (Square Root, Box-Cox, Power, etc.)**  

Non-linear transformations help **stabilize variance**, **reduce skewness**, and **improve model performance** in machine learning. These include **square root**, **Box-Cox**, and **power transformations**.

---

## **1. Square Root Transformation**
🔹 **What is it?**  
The **square root transformation** is used for data with moderate **right skewness**.  
\[
y = \sqrt{x}
\]

🔹 **Why use it?**  
- Helps **normalize skewed data**.  
- Reduces impact of **large values** without being as aggressive as log transformation.  

🔹 **Example: Applying Square Root Transformation**
```python
import numpy as np

data = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
sqrt_data = np.sqrt(data)

print(sqrt_data)
```
**Output:**
```
[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
```
✅ **Moderately reduces skewness while keeping data interpretable.**  

---

## **2. Cube Root Transformation**
🔹 **What is it?**  
Similar to the square root but works with **negative values**.  
\[
y = x^{\frac{1}{3}}
\]

🔹 **Why use it?**  
- Works for **both negative and positive** values.  
- Less aggressive than **log transformation**.  

🔹 **Example: Applying Cube Root Transformation**
```python
data = np.array([-1000, -100, -10, 0, 10, 100, 1000])
cube_root_data = np.cbrt(data)

print(cube_root_data)
```
**Output:**  
```
[-10.  -4.64  -2.15   0.   2.15  4.64 10.]
```
✅ **Works well when data includes both positive and negative values.**  

---

## **3. Box-Cox Transformation**
🔹 **What is it?**  
A **power transformation** that **optimizes** the value of **lambda (λ)** to make data **more normal**.  

🔹 **Formula:**
\[
y(\lambda) = 
\begin{cases} 
\frac{x^\lambda - 1}{\lambda}, & \text{if } \lambda \neq 0 \\
\log(x), & \text{if } \lambda = 0
\end{cases}
\]

🔹 **Why use it?**  
- Automatically finds the best **power transformation**.  
- Works **only for positive values**.  

🔹 **Example: Applying Box-Cox**
```python
from scipy.stats import boxcox

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
boxcox_data, best_lambda = boxcox(data)

print("Transformed Data:", boxcox_data)
print("Best Lambda:", best_lambda)
```
**Output (Example values):**  
```
Transformed Data: [ 0.    0.47  0.89  1.28  1.65  2.   2.34  2.67  3.   3.32]
Best Lambda: 0.2
```
✅ **Automatically selects the best transformation (λ).**  

---

## **4. Power Transformation (Yeo-Johnson)**
🔹 **What is it?**  
Similar to Box-Cox but **works with zero and negative values**.  

🔹 **Formula:**
\[
y(\lambda) = 
\begin{cases} 
\frac{(x + 1)^\lambda - 1}{\lambda}, & x \geq 0, \lambda \neq 0 \\
\frac{-(|x| + 1)^{2 - \lambda} - 1}{2 - \lambda}, & x < 0, \lambda \neq 2
\end{cases}
\]

🔹 **Why use it?**  
- Works with **negative values**.  
- Similar effect to Box-Cox.  

🔹 **Example: Applying Yeo-Johnson**
```python
from sklearn.preprocessing import PowerTransformer

data = np.array([[-10], [-5], [0], [5], [10], [50], [100]])
pt = PowerTransformer(method='yeo-johnson')
transformed_data = pt.fit_transform(data)

print(transformed_data)
```
✅ **Great for normalizing data with positive and negative values.**  

---

## **5. When to Use Different Transformations?**
| Transformation | When to Use |
|---------------|------------|
| **Square Root** | When data is **moderately skewed** and positive. |
| **Cube Root** | When data has **both negative and positive values**. |
| **Log** | When data is **highly skewed**, strictly positive. |
| **Box-Cox** | When data is **skewed and positive** (optimizes λ). |
| **Yeo-Johnson** | When data has **both negative and positive values**. |

---

## **6. Comparing Transformations Visually**
Let's plot different transformations.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Skewed data
data = np.random.exponential(scale=2, size=1000)  

# Apply transformations
sqrt_data = np.sqrt(data)
log_data = np.log1p(data)
boxcox_data, _ = boxcox(data + 1)  # Box-Cox requires positive data

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(sqrt_data, kde=True, ax=axes[0])
axes[0].set_title("Square Root Transform")
sns.histplot(log_data, kde=True, ax=axes[1])
axes[1].set_title("Log Transform")
sns.histplot(boxcox_data, kde=True, ax=axes[2])
axes[2].set_title("Box-Cox Transform")

plt.show()
```
✅ **Visualizes how transformations change data distribution.**  

---

## **7. Transformations in Scikit-Learn Pipelines**
Easily integrate transformations into **ML models**.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('power_transform', PowerTransformer(method='yeo-johnson')),
    ('model', LinearRegression())
])

pipeline.fit(X, y)
```
✅ **Ensures transformations are applied consistently in training and prediction.**  

---

## **Key Takeaways**
| Transformation | Purpose |
|---------------|------------|
| **Square Root** | Reduces moderate skewness. |
| **Cube Root** | Works with negative and positive values. |
| **Log** | Compresses large values (for skewed data). |
| **Box-Cox** | Finds best power transformation (only positive values). |
| **Yeo-Johnson** | Like Box-Cox but works with negatives. |

---

