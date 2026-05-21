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
