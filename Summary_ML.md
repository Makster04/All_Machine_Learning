# Introduction to Machine Learning (ML)

## Supervised Learning

Supervised learning involves training a model on labeled data (input-output pairs).

### Example: Predicting House Prices

We have a dataset with features like:
- **Size** (square feet)
- **Location**
- **Number of bedrooms**
- **House Price** (label)

The model learns from past data and predicts house prices based on these features.

### Common Algorithms & Code Examples

#### **1. Linear Regression (Predicting House Prices)**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample Data
X = np.array([[1400], [1600], [1800], [2000], [2200]])  # Square footage
y = np.array([245000, 312000, 279000, 308000, 334000])  # Prices

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict a new house price
predicted_price = model.predict([[1900]])
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

#### **2. Logistic Regression (Spam Detection)**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

emails = ["Win a free lottery now!", "Hello, how are you?", "Earn $1000 fast"]
labels = [1, 0, 1]  # 1 = Spam, 0 = Not Spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = LogisticRegression()
model.fit(X, labels)

# Predict if a new email is spam
new_email = vectorizer.transform(["Claim your free gift!"])
print("Spam" if model.predict(new_email) else "Not Spam")
```

---
## Unsupervised Learning

Unsupervised learning involves discovering patterns in unlabeled data.

### Example: Customer Segmentation

An e-commerce business groups customers based on behavior:
- **Frequent buyers**
- **Seasonal shoppers**
- **Discount seekers**

### Common Algorithms & Code Examples

#### **3. K-Means Clustering (Customer Segmentation)**
```python
from sklearn.cluster import KMeans
import numpy as np

# Sample Data: [Spending Score, Annual Income]
X = np.array([[20, 30000], [40, 50000], [60, 70000], [80, 90000], [100, 110000]])

# Train Model
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Predict Cluster for a New Customer
new_customer = np.array([[50, 60000]])
print(f"Cluster: {kmeans.predict(new_customer)[0]}")
```

#### **4. Principal Component Analysis (Dimensionality Reduction)**
```python
from sklearn.decomposition import PCA
import numpy as np

# Sample Data (5D space)
X = np.random.rand(10, 5)

# Reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced)
```

---
## Reinforcement Learning (RL)

Reinforcement learning (RL) involves learning through interactions with an environment using rewards and penalties.

### Example: Teaching an AI to Play a Game

An AI learns by trial and error, receiving **rewards for success** and **penalties for failure**.

### Common Algorithms & Code Examples

#### **5. Q-Learning (Solving a Maze)**
```python
import numpy as np
import random

# Define environment (3x3 grid)
Q_table = np.zeros((3, 3))  # State-Action table

def get_action(state):
    return random.choice([0, 1])  # Random action: 0=Left, 1=Right

def update_Q(state, action, reward):
    Q_table[state, action] += 0.1 * (reward + 0.9 * np.max(Q_table[state]) - Q_table[state, action])

# Simulated learning process
for episode in range(100):
    state = random.randint(0, 2)
    action = get_action(state)
    reward = 1 if action == 1 else -1  # Reward for moving right
    update_Q(state, action, reward)

print("Trained Q-Table:")
print(Q_table)
```

#### **6. Deep Q-Network (DQN) with TensorFlow (Atari Games)**
```python
import tensorflow as tf
from tensorflow import keras

# Create a simple neural network for deep reinforcement learning
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(2, activation='linear')  # Two possible actions
])

model.compile(optimizer='adam', loss='mse')
print(model.summary())
```

---
## Key Terms in Machine Learning

| Term | Meaning & Example |
|------|------------------|
| **Feature** | Input variables used for prediction (e.g., house size 🏠). |
| **Label (Target)** | The output we want to predict (e.g., house price 💰). |
| **Training Set** | Data used to train the model. |
| **Test Set** | Data used to evaluate model performance. |
| **Overfitting** | Model memorizes training data too well but fails on new data. |
| **Underfitting** | Model is too simple to capture patterns. |
| **Bias-Variance Tradeoff** | Balancing model complexity to avoid overfitting & underfitting. |


## Mathematical Equations in Machine Learning

### 1. Linear Regression Equation

Linear regression models the relationship between dependent and independent variables:

$$\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
\]$$

where:
- $$\( y \)$$ = Predicted value
- $$\( x_i \)$$ = Features
- $$\( \beta_i \)$$ = Coefficients (weights)
- $$\( \epsilon \)$$ = Error term

### 2. Logistic Regression (for Binary Classification)

$$\[
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}
\]$$

where $$\( P(Y=1|X) \)$$ represents the probability of class 1.

### 3. Cost Function for Linear Regression (Mean Squared Error - MSE)

$$\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
\]$$

where:
- $$\( m \)$$ = Number of training samples
- $$\( h_{\theta}(x) \)$$ = Predicted value
- $$\( y \)$$ = Actual value

### 4. Gradient Descent Update Rule

$$\[
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
\]$$

where $$\( \alpha \)$$ is the learning rate.

## Python Code for Machine Learning

### 1. Linear Regression using Scikit-Learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target with noise

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, label="Actual data")
plt.plot(X_test, y_pred, color='red', label="Regression line")
plt.legend()
plt.show()
```

### 2. Logistic Regression for Classification

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Binary classification (select only two classes)
X, y = X[y != 2], y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### 3. K-Means Clustering

```python
from sklearn.cluster import KMeans
import seaborn as sns

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2) * 10  # Random 2D data points

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plot clusters
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="viridis", s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200)
plt.title("K-Means Clustering")
plt.show()
```

### 4. Neural Network with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X.sum(axis=1) > 2.5).astype(int)  # Binary classification

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
```

## Conclusion

Machine Learning provides powerful tools for data analysis and prediction, with applications in healthcare, finance, robotics, and more. Understanding key terms, equations, and implementing models in Python can help in mastering ML concepts and applying them to real-world problems.
