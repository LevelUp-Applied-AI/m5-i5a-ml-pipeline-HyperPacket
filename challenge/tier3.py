"""
Tier 3: Custom Cross-Validation Engine
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score

def custom_stratified_cv(X, y, model, k=5, random_state=42):
    """
    Perform stratified k-fold cross-validation from scratch.
    
    Args:
        X (pd.DataFrame or np.ndarray): Features
        y (pd.Series or np.ndarray): Target
        model: Sklearn-compatible model
        k (int): Number of folds
        random_state: Random seed for shuffling
        
    Returns:
        np.ndarray: Array of k scores (accuracy)
    """
    X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.array(X)
    y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)
    
    np.random.seed(random_state)
    
    classes = np.unique(y_arr)
    class_indices = [np.where(y_arr == c)[0] for c in classes]
    
    for idxs in class_indices:
        np.random.shuffle(idxs)
    
    # Generate array distribution logic correctly accommodating uneven chunks
    class_folds = [np.array_split(idxs, k) for idxs in class_indices]
    
    overall_folds = []
    for fold_num in range(k):
        val_idx = np.concatenate([c_folds[fold_num] for c_folds in class_folds])
        np.random.shuffle(val_idx)
        overall_folds.append(val_idx)
        
    scores = []
    
    for fold_num in range(k):
        val_idx = overall_folds[fold_num]
        
        # The training elements are everything except the validation folder
        train_idx = np.concatenate([overall_folds[i] for i in range(k) if i != fold_num])
        
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]
        
        iter_model = clone(model)
        iter_model.fit(X_train, y_train)
        preds = iter_model.predict(X_val)
        
        scores.append(accuracy_score(y_val, preds))
        
    return np.array(scores)

if __name__ == "__main__":
    X = np.random.randn(200, 5)
    y = np.array([0]*160 + [1]*40)
    model = DummyClassifier(strategy="most_frequent")
    
    custom_scores = custom_stratified_cv(X, y, model, k=5, random_state=42)
    
    sk_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sk_scores = cross_val_score(model, X, y, cv=sk_cv, scoring='accuracy')
    
    print("Custom CV Scores  :", custom_scores)
    print("Scikit-Learn CV   :", sk_scores)
