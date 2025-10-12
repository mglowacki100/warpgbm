import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from warpgbm import WarpGBM
import lightgbm as lgb


def compare_predictions(wgbm_probs, lgb_probs, wgbm_preds, lgb_preds, y_test, dataset_name):
    """Helper to compare and report results"""
    
    # Accuracies
    wgbm_acc = accuracy_score(y_test, wgbm_preds)
    lgb_acc = accuracy_score(y_test, lgb_preds)
    
    # Log losses
    wgbm_logloss = log_loss(y_test, wgbm_probs)
    lgb_logloss = log_loss(y_test, lgb_probs)
    
    # Agreement between models
    agreement = (wgbm_preds == lgb_preds).mean()
    
    # Probability correlation (per class)
    prob_corrs = []
    for k in range(wgbm_probs.shape[1]):
        corr = np.corrcoef(wgbm_probs[:, k], lgb_probs[:, k])[0, 1]
        prob_corrs.append(corr)
    avg_prob_corr = np.mean(prob_corrs)
    
    print(f"\n{'='*70}")
    print(f"{dataset_name} Comparison (10 estimators)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'WarpGBM':>15} {'LightGBM':>15} {'Diff':>10}")
    print(f"{'-'*70}")
    print(f"{'Accuracy':<25} {wgbm_acc:>15.4f} {lgb_acc:>15.4f} {abs(wgbm_acc - lgb_acc):>10.4f}")
    print(f"{'Log Loss':<25} {wgbm_logloss:>15.4f} {lgb_logloss:>15.4f} {abs(wgbm_logloss - lgb_logloss):>10.4f}")
    print(f"{'Prediction Agreement':<25} {agreement:>15.4f}")
    print(f"{'Avg Prob Correlation':<25} {avg_prob_corr:>15.4f}")
    print(f"{'='*70}")
    
    return {
        'wgbm_acc': wgbm_acc,
        'lgb_acc': lgb_acc,
        'acc_diff': abs(wgbm_acc - lgb_acc),
        'logloss_diff': abs(wgbm_logloss - lgb_logloss),
        'agreement': agreement,
        'prob_corr': avg_prob_corr
    }


def test_iris_comparison():
    """Compare WarpGBM and LightGBM on Iris dataset (3 classes, 150 samples)"""
    print("\n" + "="*70)
    print("TEST: Iris Dataset Comparison")
    print("="*70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Shared hyperparameters
    params = {
        'max_depth': 3,
        'n_estimators': 10,
        'learning_rate': 0.1,
        'num_bins': 32,
        'min_child_weight': 1,
    }
    
    # Train WarpGBM
    wgbm_model = WarpGBM(
        objective='multiclass',
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_bins=params['num_bins'],
        min_child_weight=params['min_child_weight'],
    )
    wgbm_model.fit(X_train, y_train)
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_bin=params['num_bins'],
        min_child_weight=params['min_child_weight'],
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Predictions
    wgbm_probs = wgbm_model.predict_proba(X_test)
    wgbm_preds = wgbm_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)
    lgb_preds = lgb_model.predict(X_test)
    
    # Compare
    results = compare_predictions(wgbm_probs, lgb_probs, wgbm_preds, lgb_preds, y_test, "Iris")
    
    # Assertions - should be very similar with 10 trees
    # With only 10 trees, allow some variance but verify high correlation
    assert results['acc_diff'] < 0.15, f"Accuracy difference too large: {results['acc_diff']}"
    assert results['logloss_diff'] < 0.4, f"Log loss difference too large: {results['logloss_diff']}"
    assert results['prob_corr'] > 0.90, f"Probability correlation too low: {results['prob_corr']}"
    
    print("\n✓ Iris test passed - predictions are close!")


def test_wine_comparison():
    """Compare on Wine dataset (3 classes, 178 samples, 13 features)"""
    print("\n" + "="*70)
    print("TEST: Wine Dataset Comparison")
    print("="*70)
    
    # Load data
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Shared hyperparameters
    params = {
        'max_depth': 4,
        'n_estimators': 10,
        'learning_rate': 0.1,
        'num_bins': 32,
        'min_child_weight': 1,
    }
    
    # Train WarpGBM
    wgbm_model = WarpGBM(
        objective='multiclass',
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_bins=params['num_bins'],
        min_child_weight=params['min_child_weight'],
    )
    wgbm_model.fit(X_train, y_train)
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_bin=params['num_bins'],
        min_child_weight=params['min_child_weight'],
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Predictions
    wgbm_probs = wgbm_model.predict_proba(X_test)
    wgbm_preds = wgbm_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)
    lgb_preds = lgb_model.predict(X_test)
    
    # Compare
    results = compare_predictions(wgbm_probs, lgb_probs, wgbm_preds, lgb_preds, y_test, "Wine")
    
    # Assertions
    assert results['acc_diff'] < 0.15, f"Accuracy difference too large: {results['acc_diff']}"
    assert results['logloss_diff'] < 0.4, f"Log loss difference too large: {results['logloss_diff']}"
    assert results['prob_corr'] > 0.85, f"Probability correlation too low: {results['prob_corr']}"
    
    print("\n✓ Wine test passed - predictions are close!")


def test_synthetic_5class_comparison():
    """Compare on synthetic 5-class problem (more complex)"""
    print("\n" + "="*70)
    print("TEST: Synthetic 5-Class Comparison")
    print("="*70)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=2,
        random_state=42,
        flip_y=0.05
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Shared hyperparameters
    params = {
        'max_depth': 5,
        'n_estimators': 10,
        'learning_rate': 0.1,
        'num_bins': 32,
        'min_child_weight': 5,
    }
    
    # Train WarpGBM
    wgbm_model = WarpGBM(
        objective='multiclass',
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_bins=params['num_bins'],
        min_child_weight=params['min_child_weight'],
    )
    wgbm_model.fit(X_train, y_train)
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=5,
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_bin=params['num_bins'],
        min_child_weight=params['min_child_weight'],
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Predictions
    wgbm_probs = wgbm_model.predict_proba(X_test)
    wgbm_preds = wgbm_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)
    lgb_preds = lgb_model.predict(X_test)
    
    # Compare
    results = compare_predictions(wgbm_probs, lgb_probs, wgbm_preds, lgb_preds, y_test, "Synthetic 5-Class")
    
    # Assertions - slightly more lenient for harder problem
    assert results['acc_diff'] < 0.15, f"Accuracy difference too large: {results['acc_diff']}"
    assert results['logloss_diff'] < 0.4, f"Log loss difference too large: {results['logloss_diff']}"
    assert results['prob_corr'] > 0.80, f"Probability correlation too low: {results['prob_corr']}"
    
    print("\n✓ Synthetic 5-class test passed - predictions are close!")


def test_binary_comparison():
    """Compare on binary classification"""
    print("\n" + "="*70)
    print("TEST: Binary Classification Comparison")
    print("="*70)
    
    # Generate binary data
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.05
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Shared hyperparameters
    params = {
        'max_depth': 4,
        'n_estimators': 10,
        'learning_rate': 0.1,
        'num_bins': 32,
        'min_child_weight': 5,
    }
    
    # Train WarpGBM
    wgbm_model = WarpGBM(
        objective='binary',
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_bins=params['num_bins'],
        min_child_weight=params['min_child_weight'],
    )
    wgbm_model.fit(X_train, y_train)
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_bin=params['num_bins'],
        min_child_weight=params['min_child_weight'],
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Predictions
    wgbm_probs = wgbm_model.predict_proba(X_test)
    wgbm_preds = wgbm_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)
    lgb_preds = lgb_model.predict(X_test)
    
    # Compare
    results = compare_predictions(wgbm_probs, lgb_probs, wgbm_preds, lgb_preds, y_test, "Binary")
    
    # Assertions
    assert results['acc_diff'] < 0.15, f"Accuracy difference too large: {results['acc_diff']}"
    assert results['logloss_diff'] < 0.4, f"Log loss difference too large: {results['logloss_diff']}"
    assert results['prob_corr'] > 0.85, f"Probability correlation too low: {results['prob_corr']}"
    
    print("\n✓ Binary test passed - predictions are close!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WarpGBM vs LightGBM Classification Comparison Suite")
    print("Testing with 10 estimators to verify implementation correctness")
    print("="*70)
    
    test_iris_comparison()
    test_wine_comparison()
    test_synthetic_5class_comparison()
    test_binary_comparison()
    
    print("\n" + "="*70)
    print("✓ ALL COMPARISON TESTS PASSED!")
    print("WarpGBM classification produces results very close to LightGBM")
    print("="*70)

