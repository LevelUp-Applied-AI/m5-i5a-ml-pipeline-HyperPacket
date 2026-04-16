"""
Tier 2: Pipeline Factory with Feature Engineering
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
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

def build_pipeline(model, numeric_features, categorical_features, use_poly=False):
    if use_poly:
        num_transformer = Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=True)),
            ("scaler", StandardScaler())
        ])
    else:
        num_transformer = StandardScaler()
        
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

def run_pipeline_factory():
    X, y = load_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "LogReg_default": LogisticRegression(C=1.0, random_state=42, max_iter=2000, class_weight="balanced"),
        "RidgeClassifier": RidgeClassifier(alpha=1.0, random_state=42, class_weight="balanced")
    }
    
    scoring = ["accuracy", "precision", "recall", "f1"]
    
    print(f"{'Model':<20} | {'Poly':<5} | {'F1 Mean':<8} | {'Recall':<8}")
    print("-" * 50)
    
    for name, clf in models.items():
        for poly_flag in [False, True]:
            pipeline = build_pipeline(clf, NUMERIC_FEATURES, CATEGORICAL_FEATURES, use_poly=poly_flag)
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv_strategy, scoring=scoring)
            f1_mean = cv_results["test_f1"].mean()
            recall_mean = cv_results["test_recall"].mean()
            print(f"{name:<20} | {str(poly_flag):<5} | {f1_mean:.4f}   | {recall_mean:.4f}")

if __name__ == "__main__":
    run_pipeline_factory()
