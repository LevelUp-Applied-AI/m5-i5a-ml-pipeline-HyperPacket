import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def load_data(filepath="../data/telecom_churn.csv"):
    df = pd.read_csv(filepath)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    X = df.drop(columns=["churned"])
    y = df["churned"]
    return X, y

def build_pipeline():
    NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen",
                        "has_partner", "has_dependents"]

    CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                            "payment_method"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    
    return pipeline

def main():
    # 1. Load Data
    X, y = load_data()
    
    # 2. Build model pipeline
    pipeline = build_pipeline()
    
    # 3. Setup Stratified Cross-Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We use F1-score as the metric due to class imbalance (churn rate is roughly 16% in this dataset type)
    # Accuracy would be misleading as a naive classifier easily achieves high accuracy by predicting the majority class.
    scoring_metric = "f1"
    
    # 4. Compute Learning Curve
    print("Computing learning curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv_strategy,
        scoring=scoring_metric,
        n_jobs=-1,
        random_state=42
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print("\n--- Learning Curve Results ---")
    for i, size in enumerate(train_sizes):
        print(f"Train Size: {int(size)} | Train Score: {train_mean[i]:.4f} +/- {train_std[i]:.4f} | Val Score: {val_mean[i]:.4f} +/- {val_std[i]:.4f}")
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, val_mean, 'o-', color="orange", label="Validation (Cross-validation) Score")
    
    # Add shaded regions for +/- 1 standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")
    
    plt.title("Learning Curve (Logistic Regression)")
    plt.xlabel("Number of Training Examples")
    plt.ylabel(f"Score ({scoring_metric.upper()})")
    plt.legend(loc="best")
    plt.grid(True)
    
    # Save the plot
    output_filename = "learning_curve_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    main()
