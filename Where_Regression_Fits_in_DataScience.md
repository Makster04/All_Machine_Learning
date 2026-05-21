# Where Regression Fits in Data Science

```
Data Science
│
└── Machine Learning
    │
    ├── Supervised Learning          ← YOU ARE HERE
    │   ├── Regression               ← specifically this
    │   │   ├── Linear Regression
    │   │   ├── Polynomial Regression
    │   │   └── Ridge / Lasso
    │   │
    │   └── Classification
    │       ├── Logistic Regression
    │       └── Decision Trees
    │
    └── Unsupervised Learning
        ├── Clustering (k-means)
        └── Dimensionality Reduction (t-SNE)
```

---

## The 3 stages your key terms belong to

### 1. Data Preprocessing
> Steps taken *before* fitting the model.

- **Dummy Variables** — Categorical variables transformed into binary variables (0 or 1) for use in regression models.
- **One-Hot Encoding** — A technique to convert categorical variables into binary variables, where each category is represented by a separate column.

---

### 2. Model Fitting
> The building blocks of the regression equation itself.

- **Dependent Variable (Y)** — The outcome variable that the model aims to predict.
- **Independent Variables (X)** — The input variables used to predict the dependent variable.
- **Intercept (β₀)** — The value of the dependent variable when all independent variables are equal to zero.
- **Coefficients (β)** — The estimated values representing the relationship between each independent variable and the dependent variable.

---

### 3. Model Evaluation
> Metrics used *after* fitting to measure how well the model did.

- **R-Squared** — The proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Adjusted R-Squared** — A modified version of R-Squared that adjusts for the number of predictors, providing a more accurate measure when there are multiple predictors.
- **Mean Absolute Error (MAE)** — The average of the absolute errors between the predicted and actual values.
- **Root Mean Squared Error (RMSE)** — The square root of the average of the squared differences between predicted and actual values.

**Dataset (what we're working with):**

| House | Size (sqft) | Bedrooms | Neighborhood |  Price  |
|-------|-------------|----------|--------------|---------|
| A     | 1,500       | 3        | Suburban     | $300,000 |
| B     | 2,000       | 4        | Urban        | $450,000 |
| C     | 1,200       | 2        | Rural        | $200,000 |
| D     | 1,800       | 3        | Urban        | $400,000 |

---

**Dependent Variable (Y):** `Price` — the thing we're trying to predict. Every other column exists to help us guess this one.

**Independent Variables (X):** `Size`, `Bedrooms`, and `Neighborhood` — the inputs we feed the model to predict price.

**Intercept (β₀):** If size = 0, bedrooms = 0, and neighborhood = 0, the model might output `$50,000`. That baseline value is the intercept — not always realistic, but mathematically necessary.

**Coefficients (β):** After training, the model might learn:
- β₁ (Size) = `$150 per sqft`
- β₂ (Bedrooms) = `$10,000 per bedroom`

So: `Price = 50,000 + (150 × Size) + (10,000 × Bedrooms)`

---

**Dummy Variables:** `Neighborhood` is categorical (Suburban, Urban, Rural), so we can't use it as-is. We pick one category as the *baseline* (Rural = 0) and create binary flags for the rest:

| House | is_Suburban | is_Urban |
|-------|-------------|----------|
| A     | 1           | 0        |
| B     | 0           | 1        |
| C     | 0           | 0        |
| D     | 0           | 1        |

Rural is implied when both columns = 0. This is the dummy variable approach — **k-1 columns** for k categories.

**One-Hot Encoding:** Similar idea, but you keep *all* categories as separate columns with no baseline dropped:

| House | is_Suburban | is_Urban | is_Rural |
|-------|-------------|----------|----------|
| A     | 1           | 0        | 0        |
| B     | 0           | 1        | 0        |
| C     | 0           | 0        | 1        |
| D     | 0           | 1        | 0        |

Each row has exactly one `1`. Used more often in ML models (vs. regression where you drop one to avoid multicollinearity).

---

**Now the model makes predictions:**

| House | Actual Price | Predicted Price | Error       | |Error| | Error² |
|-------|-------------|-----------------|-------------|---------|--------|
| A     | $300,000    | $320,000        | −$20,000    | $20,000 | 400,000,000 |
| B     | $450,000    | $430,000        | +$20,000    | $20,000 | 400,000,000 |
| C     | $200,000    | $180,000        | +$20,000    | $20,000 | 400,000,000 |
| D     | $400,000    | $370,000        | +$30,000    | $30,000 | 900,000,000 |

**MAE:** Average of |Error| column → `(20k + 20k + 20k + 30k) / 4 = $22,500`. Easy to interpret — on average, predictions are off by $22,500.

**RMSE:** √(average of Error² column) → `√(525,000,000) ≈ $22,912`. Similar to MAE here, but RMSE punishes House D's larger $30k error more heavily because of the squaring step.

---

**R-Squared:** Suppose the model explains 85% of the variation in house prices — meaning size, bedrooms, and neighborhood account for most of why prices differ. R² = **0.85**.

**Adjusted R-Squared:** If you then add a useless variable like "house ID number," plain R² might creep up to 0.86 even though the new variable adds nothing. Adjusted R² would stay near **0.85**, penalizing you for adding a predictor that didn't earn its place.

---

The key takeaway: MAE and RMSE tell you *how far off* your predictions are in dollar terms, while R² and Adjusted R² tell you *how much of the story* your model is explaining.



# Classification in Data Science

## Where It Fits

```
Data Science
│
└── Machine Learning
    │
    ├── Supervised Learning          ← YOU ARE HERE
    │   ├── Regression
    │   │   ├── Linear Regression
    │   │   ├── Polynomial Regression
    │   │   └── Ridge / Lasso
    │   │
    │   └── Classification           ← specifically this
    │       ├── Logistic Regression  ← and these two
    │       └── Decision Trees       ← and these two
    │
    └── Unsupervised Learning
        ├── Clustering (k-means)
        └── Dimensionality Reduction (t-SNE)
```

Classification is also a **Supervised Learning** method — but instead of predicting a *number* like regression does, it predicts a *category* or *label*.

| | Regression | Classification |
|---|---|---|
| Output | A number (e.g. $300,000) | A category (e.g. Spam / Not Spam) |
| Example | Predict house price | Predict if an email is spam |
| Metrics | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |

---

## The Dataset

Throughout this guide, we use an **email spam detection** dataset as a running example:

| Email | Word Count | Has Links | Sender Known | Label     |
|-------|------------|-----------|--------------|-----------|
| A     | 300        | Yes       | No           | Spam      |
| B     | 150        | No        | Yes          | Not Spam  |
| C     | 500        | Yes       | No           | Spam      |
| D     | 200        | No        | Yes          | Not Spam  |
| E     | 450        | Yes       | No           | Spam      |
| F     | 100        | No        | Yes          | Not Spam  |

The goal: **predict whether an email is Spam or Not Spam** using word count, links, and sender.

---

## Stage 1 — Data Preprocessing
> Steps taken *before* fitting the model.

Just like in regression, categorical columns need to be converted to numbers before the model can use them.

### Encoding the Label (Y)
The outcome itself is categorical — Spam or Not Spam. We encode it as a binary number:
- `Spam = 1`
- `Not Spam = 0`

| Email | Label    | Encoded |
|-------|----------|---------|
| A     | Spam     | 1       |
| B     | Not Spam | 0       |
| C     | Spam     | 1       |
| D     | Not Spam | 0       |

### One-Hot Encoding the Inputs (X)
`Has Links` and `Sender Known` are also categorical (Yes/No). We convert them to binary columns:

| Email | Word Count | Has Links | Sender Known |
|-------|------------|-----------|--------------|
| A     | 300        | 1         | 0            |
| B     | 150        | 0         | 1            |
| C     | 500        | 1         | 0            |
| D     | 200        | 0         | 1            |

Now every column is a number — the model can work with this.

---

## Stage 2 — Model Fitting
> How each classification model learns from the data.

There are two models on the tree: **Logistic Regression** and **Decision Trees**. They solve the same problem but in very different ways.

---

### Model A — Logistic Regression

Logistic Regression works similarly to Linear Regression, but instead of predicting a number, it predicts a **probability between 0 and 1** using an S-shaped curve (called the sigmoid function).

```
P(Spam) = sigmoid(β₀ + β₁×WordCount + β₂×HasLinks + β₃×SenderKnown)
```

After training, the model might learn:
- β₀ (Intercept) = `−2.0` (baseline lean toward Not Spam)
- β₁ (Word Count) = `+0.005` (more words → slightly more likely spam)
- β₂ (Has Links)  = `+1.8`  (having links strongly increases spam probability)
- β₃ (Sender Known) = `−2.5` (known sender strongly decreases spam probability)

**Example — Email A** (300 words, has links, sender unknown):
```
raw score = −2.0 + (0.005×300) + (1.8×1) + (−2.5×0)
raw score = −2.0 + 1.5 + 1.8 + 0
raw score = 1.3

P(Spam) = sigmoid(1.3) ≈ 0.79
```
> The model is **79% confident** Email A is spam.

### The Threshold
The model outputs a probability — you decide the cutoff for calling something Spam.

- Default threshold = `0.5`
- `P(Spam) ≥ 0.5` → predicted **Spam**
- `P(Spam) < 0.5` → predicted **Not Spam**

You can raise or lower this threshold depending on your situation (more on this in Stage 3).

### The Decision Boundary
The threshold creates a **decision boundary** — the line that separates the two classes. In our example, emails above a certain probability score get labeled Spam; everything below gets labeled Not Spam.

---

### Model B — Decision Trees

A Decision Tree doesn't use probabilities or equations. Instead, it asks a series of **yes/no questions** about the features and follows a path to a final answer — like a flowchart.

```
                    [Has Links?]
                   /            \
                Yes              No
                /                  \
      [Sender Known?]         → NOT SPAM
       /         \
     Yes          No
      |            |
  NOT SPAM       SPAM
```

**Example — Email A** (has links, sender unknown):
1. Has Links? → **Yes** → go left
2. Sender Known? → **No** → go right
3. Result: → **SPAM** ✓

**Example — Email B** (no links, sender known):
1. Has Links? → **No** → go right
2. Result: → **NOT SPAM** ✓

### Information Gain
The tree decides *which question to ask first* using **Information Gain** — a measure of how well a feature splits the data into clean groups. A good split separates spam from not-spam as cleanly as possible. Higher information gain = better split = asked earlier in the tree.

In our dataset, `Has Links` has the highest information gain because it almost perfectly divides the emails — so it becomes the root (first) question.

---

## Stage 3 — Model Evaluation
> Metrics used *after* fitting to measure how well the model did.

### Predictions vs. Reality

After running all emails through the model, we get:

| Email | Actual    | Predicted | Correct? |
|-------|-----------|-----------|----------|
| A     | Spam      | Spam      | ✓        |
| B     | Not Spam  | Not Spam  | ✓        |
| C     | Spam      | Spam      | ✓        |
| D     | Not Spam  | Spam      | ✗        |
| E     | Spam      | Spam      | ✓        |
| F     | Not Spam  | Not Spam  | ✓        |

Email D was wrongly predicted as Spam — let's break down what that means.

---

### The Confusion Matrix

A confusion matrix organizes all predictions into four buckets:

|                     | Predicted: Spam | Predicted: Not Spam |
|---------------------|-----------------|---------------------|
| **Actual: Spam**    | TP = 3          | FN = 0              |
| **Actual: Not Spam**| FP = 1          | TN = 2              |

- **True Positive (TP):** Model said Spam, actually was Spam → 3 emails (A, C, E)
- **True Negative (TN):** Model said Not Spam, actually was Not Spam → 2 emails (B, F)
- **False Positive (FP):** Model said Spam, actually was Not Spam → 1 email (D) ← the mistake
- **False Negative (FN):** Model said Not Spam, actually was Spam → 0 emails

---

### Accuracy
The simplest metric — what percentage of all predictions were correct?

```
Accuracy = (TP + TN) / Total = (3 + 2) / 6 = 0.833 = 83.3%
```

> The model got 5 out of 6 emails right.

⚠️ Accuracy can be misleading with imbalanced data. If 95% of emails are Not Spam, a model that *always* predicts Not Spam would score 95% accuracy — but it would be completely useless.

---

### Precision
Of all the emails the model *called* Spam, how many actually were?

```
Precision = TP / (TP + FP) = 3 / (3 + 1) = 0.75 = 75%
```

> When the model flags an email as Spam, it's right **75% of the time**.
> High precision = fewer false alarms.

---

### Recall (Sensitivity)
Of all the emails that *actually were* Spam, how many did the model catch?

```
Recall = TP / (TP + FN) = 3 / (3 + 0) = 1.0 = 100%
```

> The model caught **every single spam email** — it missed none.
> High recall = fewer missed positives.

---

### The Precision vs. Recall Trade-off

You can't always have both. Lowering the threshold catches more spam (higher recall) but also flags more legitimate emails as spam (lower precision). Raising the threshold is more careful (higher precision) but misses more actual spam (lower recall).

| Scenario | What It Means | When It Matters |
|----------|---------------|-----------------|
| High Recall, Low Precision | Catches almost all spam but has many false alarms | Medical diagnosis — missing a real case is costly |
| High Precision, Low Recall | Only flags things it's very sure about, but misses some | Legal evidence — a false accusation is very costly |
| High Recall, High Precision | The ideal — catches spam and rarely makes mistakes | The goal every model aims for |
| Low Recall, Low Precision | Missing positives and making false accusations | A poorly trained model |

---

### F1-Score
A single number that balances precision and recall. Useful when you care about both equally.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 2 × (0.75 × 1.0) / (0.75 + 1.0)
F1 = 1.5 / 1.75 ≈ 0.857 = 85.7%
```

> The F1-Score of **85.7%** reflects strong recall pulling up a lower precision.

---

### AUC-ROC
Instead of evaluating at one threshold, AUC-ROC evaluates the model *across all possible thresholds* and summarizes it as a single score.

- **AUC = 1.0** → Perfect model
- **AUC = 0.5** → No better than random guessing
- **AUC = 0.0** → Perfectly wrong every time

> AUC-ROC is especially useful when comparing two different models — the one with the higher AUC is better at separating the two classes regardless of what threshold you pick.

---

### Log-Loss (Cross-Entropy Loss)
Used specifically with Logistic Regression. Instead of just checking if the prediction was right or wrong, it checks **how confident** the model was.

A model that says "I'm 99% sure this is spam" but is wrong gets penalized *far more* than one that says "I'm 55% sure this is spam" and is wrong.

```
Lower Log-Loss = better calibrated probability predictions
```

> This is why Gradient Descent optimizes Log-Loss during Logistic Regression training — it pushes the model to be both correct *and* appropriately confident.

---

## Summary

### Logistic Regression vs. Decision Trees

| | Logistic Regression | Decision Tree |
|---|---|---|
| How it works | Learns an equation with coefficients | Learns a flowchart of yes/no questions |
| Output | A probability (e.g. 0.79) | A class (e.g. Spam) |
| Decision boundary | A smooth curve | A series of hard splits |
| Best for | Linear relationships between features and outcome | Non-linear relationships, easier to explain |

### Evaluation Metrics at a Glance

| Metric | What It Tells You | Ideal Value |
|--------|-------------------|-------------|
| Accuracy | % of all predictions that were correct | As high as possible |
| Precision | Of predicted positives, how many were actually positive | As high as possible |
| Recall | Of actual positives, how many did the model catch | As high as possible |
| F1-Score | Balance between precision and recall | As high as possible |
| AUC-ROC | How well the model separates classes at any threshold | Close to 1.0 |
| Log-Loss | How confident and correct the probability predictions are | As low as possible |

**The bottom line:**
- Use **Accuracy** for a quick check — but not with imbalanced data.
- Use **Precision** when false positives are costly (e.g. wrongly flagging a legitimate email).
- Use **Recall** when false negatives are costly (e.g. missing a cancer diagnosis).
- Use **F1-Score** when you need to balance both.
- Use **AUC-ROC** when comparing models.
- Use **Log-Loss** when the probability score itself matters, not just the final label.

