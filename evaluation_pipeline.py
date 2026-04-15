"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    df = pd.read_csv(filepath)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    X = df.drop(columns=["churned"])
    y = df["churned"]
    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )


def define_models():
    """Define the 5 model configurations to compare.

    Two dummy baselines are included to teach two different lessons:
    most_frequent demonstrates the accuracy inflation problem on imbalanced
    data; stratified shows what random guessing in proportion to class
    frequencies looks like, so F1 carries meaningful signal when comparing.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    preprocessor = build_preprocessor()
    
    return {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=1.0, random_state=42, max_iter=1000, class_weight="balanced"))
        ]),
        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=0.1, penalty="l1", solver="saga", random_state=42, max_iter=1000, class_weight="balanced"))
        ]),
        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RidgeClassifier(alpha=1.0, random_state=42, class_weight="balanced"))
        ]),
        "Dummy_most_frequent": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ]),
        "Dummy_stratified": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="stratified", random_state=42))
        ])
    }


def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results.

    Args:
        models: Dictionary of {name: Pipeline}.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of folds.
        random_state: Random seed.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean.
    """
    results_list = []
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for name, pipeline in models.items():
        cv_results = cross_validate(pipeline, X, y, cv=cv_strategy, scoring=scoring)
        results_list.append({
            "model": name,
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "precision_mean": cv_results["test_precision"].mean(),
            "recall_mean": cv_results["test_recall"].mean(),
            "f1_mean": cv_results["test_f1"].mean()
        })
        
    return pd.DataFrame(results_list)


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """Train a pipeline on full training data and evaluate on the held-out test set.

    Use this on the best model from Task 4 as a final sanity check — the
    test-set metrics should be close to the CV estimates if the model
    generalizes. If they diverge substantially, the CV estimates were
    optimistic and you should investigate.

    Args:
        pipeline: An unfitted sklearn Pipeline (one entry from define_models).
        X_train, X_test: Feature DataFrames (train and held-out test).
        y_train, y_test: Target Series (train and held-out test).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def recommend_model(results_df):
    """Print a recommendation based on the results.

    Args:
        results_df: DataFrame from evaluate_models.
    """
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")


if __name__ == "__main__":
    data = load_and_prepare()
    if data is not None:
        X, y = data
        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")

        # Create 80/20 train/test split. The test set is held out for the
        # final evaluation in Task 5 — do not use it during cross-validation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

        models = define_models()
        if models:
            # Task 4: cross-validation on training data only
            results = evaluate_models(models, X_train, y_train)
            if results is not None:
                recommend_model(results)

                # Task 5: final evaluation on the held-out test set.
                real_models = results[~results['model'].str.contains('Dummy')]
                best_model_name = real_models.sort_values(by='f1_mean', ascending=False).iloc[0]['model']
                
                print(f"\nEvaluating Best Model: {best_model_name}")
                best_pipeline = models[best_model_name]
                
                test_metrics = final_evaluation(best_pipeline, X_train, X_test, y_train, y_test)
                
                print("\n=== Final Test Set Metrics ===")
                for metric, value in test_metrics.items():
                    print(f"{metric.capitalize()}: {value:.4f}")

"""
=== RECOMMENDATION ===
I recommend the RidgeClassifier model, as it achieved the highest mean F1-score (~0.341) during cross-validation and generalized successfully to the unseen test set with an F1-score of 0.381. Accuracy alone is a misleading metric for this imbalanced dataset because churn detection carries asymmetric business costs; the most-frequent dummy baseline clearly proves this by achieving an 83.75% accuracy while failing to pinpoint a single real churner. The RidgeClassifier explicitly trades general accuracy (dropping to ~61.1%) to achieve a high recall (~62.1%), correctly identifying the majority of churning customers at an acceptable tradeoff in precision (~23.6%). Compared to the stratified dummy (F1 ~0.173), our chosen model captures actionable churners at roughly twice the rate of random guessing, confirming that it has extracted structural meaning from the features. While this model demonstrates real value, a 0.38 Test F1 indicates we are likely hitting the structural performance ceiling for linear models on tabular data, suggesting that tree-based classifiers should be explored before committing to a final production deployment. Ultimately, our final hold-out test set metrics fully validate our robust CV selection technique, as the test scores are fully aligned with the CV estimates and show no signs of overfitting or selection optimism.
"""
