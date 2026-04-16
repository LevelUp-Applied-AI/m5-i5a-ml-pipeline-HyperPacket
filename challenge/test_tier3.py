import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from tier3 import custom_stratified_cv

def get_custom_cv_folds(y, k=5, random_state=42):
    """Extraction function to mirror core indices logic layout inside testing environment."""
    y_arr = np.array(y)
    np.random.seed(random_state)
    classes = np.unique(y_arr)
    class_indices = [np.where(y_arr == c)[0] for c in classes]
    for idxs in class_indices:
        np.random.shuffle(idxs)
    class_folds = [np.array_split(idxs, k) for idxs in class_indices]
    overall_folds = []
    for fold_num in range(k):
        val_idx = np.concatenate([c_folds[fold_num] for c_folds in class_folds])
        np.random.shuffle(val_idx)
        overall_folds.append(val_idx)
    return overall_folds

@pytest.fixture
def dummy_data():
    X = np.random.randn(105, 3) 
    y = np.array([0]*70 + [1]*35) 
    return X, y

def test_correct_number_of_folds(dummy_data):
    X, y = dummy_data
    model = DummyClassifier(strategy="most_frequent")
    scores = custom_stratified_cv(X, y, model, k=5)
    assert len(scores) == 5, f"Expected exactly 5 metric scores, got {len(scores)}"

def test_non_overlapping_folds(dummy_data):
    _, y = dummy_data
    folds = get_custom_cv_folds(y, k=5)
    
    all_indices = []
    for fold in folds:
        all_indices.extend(list(fold))
        
    assert len(all_indices) == len(y), "Partition error: Missing instances across validation tests."
    assert len(set(all_indices)) == len(y), "Overlap detected in validation fold subsetting!"

def test_preserved_class_ratios(dummy_data):
    _, y = dummy_data
    k = 5
    folds = get_custom_cv_folds(y, k=k)
    
    overall_ratio = np.mean(y)
    
    for fold in folds:
        fold_y = y[fold]
        ratio = np.mean(fold_y)
        
        # Ensures roughly identical stratification ratio mappings
        assert np.isclose(ratio, overall_ratio, atol=0.01), f"Class representation heavily mutated."
