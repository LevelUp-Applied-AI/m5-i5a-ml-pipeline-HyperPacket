# Challenge Extension Results

## Tier 1: Per-Class Analysis
After running out-of-fold predictions using `cross_val_predict`, we can evaluate how each model specifically impacts the minority class (Churned).

### Findings:
- **Best Minority Class Model:** Both `RidgeClassifier` and `LogReg_L1` perform similarly, but `LogReg_L1` achieved the absolute highest recall (0.6256) on the minority class compared to default Logistic Regression (0.6000). Note that `RidgeClassifier` maintained slightly better structural F1 due to a better precision tradeoff.
- **L1 Regularization Effect:** Switching from default LogisticRegression to L1-regularized penalty focuses the mathematical weights on the most critical predictors, completely zeroing out noise. This structural difference meaningfully boosts our recall (from 60.00% to 62.56%) for the churn class, confirming that removing noisy features explicitly helped the model isolate the true structural drivers of churn better.

## Tier 2: Pipeline Factory with Feature Engineering
We added `PolynomialFeatures(degree=2, interaction_only=True)` to assess whether higher-dimensional combinations of numeric variables extract hidden patterns.

### Findings:
- **Did feature engineering improve the model?** No, adding polynomial interaction features strictly harmed both evaluation metrics (Mean F1 fell from `0.3336 -> 0.3303` for LogReg_default and `0.3411 -> 0.3283` for Ridge). Recall also dropped significantly across the board (from 62% dropping to 58.4%).
- **Bias-Variance Tradeoff Impact:** Expanding our dimensionality by constructing polynomial pairs increased our model's complexity severely (adding numerous collinear variables). This tipped the bias-variance tradeoff towards **high variance**, meaning our models began fitting strictly to the training split statistical noise instead of actual signal, thus reducing out-of-fold generalizability when subjected to CV subsets. Linear models generally struggle when confronted mechanically with extreme feature ballooning absent aggressive, fine-tuned regularization strategies.
