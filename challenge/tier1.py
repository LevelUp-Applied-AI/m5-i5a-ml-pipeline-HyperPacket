"""
Tier 1: Per-Class Analysis
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges", "num_support_calls", "senior_citizen", "has_partner", "has_dependents"]
CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service", "payment_method"]

def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    X = df.drop(columns=["churned"])
    y = df["churned"]
    return X, y

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def define_models():
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
        ])
    }

def run_per_class_analysis():
    X, y = load_data()
    # Isolate training set
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = define_models()
    
    for name, model in models.items():
        print(f"\n=== Model: {name} ===")
        # Get out-of-fold predictions
        y_pred = cross_val_predict(model, X_train, y_train, cv=cv_strategy)
        print(classification_report(y_train, y_pred, target_names=["Not Churned", "Churned"], digits=4))

if __name__ == "__main__":
    run_per_class_analysis()
