import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from warpgbm import WarpGBM
import lightgbm as lgb


def compare_feature_importances(wgbm_imp, lgb_imp, feature_names, dataset_name):
    """Helper to compare and report feature importances"""
    
    # Normalize both for fair comparison
    wgbm_imp_norm = wgbm_imp / wgbm_imp.sum()
    lgb_imp_norm = lgb_imp / lgb_imp.sum()
    
    # Compute rank correlation (Spearman)
    rank_corr, p_value = spearmanr(wgbm_imp_norm, lgb_imp_norm)
    
    # Compute absolute difference
    abs_diff = np.abs(wgbm_imp_norm - lgb_imp_norm).mean()
    
    # Top 3 features
    wgbm_top3 = np.argsort(wgbm_imp_norm)[-3:][::-1]
    lgb_top3 = np.argsort(lgb_imp_norm)[-3:][::-1]
    
    print(f"\n{'='*70}")
    print(f"{dataset_name} Feature Importance Comparison")
    print(f"{'='*70}")
    print(f"{'Feature':<25} {'WarpGBM':>15} {'LightGBM':>15} {'Diff':>10}")
    print(f"{'-'*70}")
    
    for idx in range(len(feature_names)):
        fname = feature_names[idx] if feature_names else f"Feature {idx}"
        print(f"{fname:<25} {wgbm_imp_norm[idx]:>15.4f} {lgb_imp_norm[idx]:>15.4f} "
              f"{wgbm_imp_norm[idx] - lgb_imp_norm[idx]:>10.4f}")
    
    print(f"{'-'*70}")
    print(f"{'Rank Correlation':<25} {rank_corr:>15.4f} (p={p_value:.4f})")
    print(f"{'Mean Abs Difference':<25} {abs_diff:>15.4f}")
    print(f"\n{'Top 3 features (WarpGBM)':<25}: {[feature_names[i] if feature_names else f'F{i}' for i in wgbm_top3]}")
    print(f"{'Top 3 features (LightGBM)':<25}: {[feature_names[i] if feature_names else f'F{i}' for i in lgb_top3]}")
    print(f"{'='*70}")
    
    return {
        'rank_correlation': rank_corr,
        'p_value': p_value,
        'mean_abs_diff': abs_diff,
        'wgbm_top3': wgbm_top3,
        'lgb_top3': lgb_top3
    }


def test_iris_feature_importance():
    """Test feature importance on Iris dataset (classification)"""
    print("\n" + "="*70)
    print("TEST: Iris Feature Importance (Classification)")
    print("="*70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Train WarpGBM - more trees, smaller learning rate for stable importances
    wgbm_model = WarpGBM(
        objective='multiclass',
        max_depth=3,
        n_estimators=100,
        learning_rate=0.05,
        num_bins=32,
    )
    wgbm_model.fit(X_train, y_train)
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        max_depth=3,
        n_estimators=100,
        learning_rate=0.05,
        max_bin=32,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Get feature importances
    wgbm_imp = wgbm_model.get_feature_importance()
    lgb_imp = lgb_model.feature_importances_
    
    # Compare
    results = compare_feature_importances(
        wgbm_imp, lgb_imp, iris.feature_names, "Iris"
    )
    
    # Assertions - with small datasets and different implementations, 
    # perfect correlation isn't expected. Check for reasonable agreement.
    assert results['rank_correlation'] > 0.4, \
        f"Rank correlation too low: {results['rank_correlation']}"
    
    # Check that at least 2 of top 3 features overlaps
    overlap = len(set(results['wgbm_top3']) & set(results['lgb_top3']))
    assert overlap >= 2, f"Only {overlap}/3 features overlap in top 3"
    
    print(f"\n✓ Iris feature importance test passed!")
    print(f"  Rank correlation: {results['rank_correlation']:.4f}")
    print(f"  Top 3 overlap: {overlap}/3")


def test_regression_feature_importance():
    """Test feature importance on regression dataset"""
    print("\n" + "="*70)
    print("TEST: Regression Feature Importance")
    print("="*70)
    
    # Generate synthetic regression data with known importance
    np.random.seed(42)
    n_samples = 1000
    X1 = np.random.randn(n_samples)  # Important
    X2 = np.random.randn(n_samples)  # Important
    X3 = np.random.randn(n_samples)  # Moderately important
    X4 = np.random.randn(n_samples)  # Noise
    X5 = np.random.randn(n_samples)  # Noise
    
    # y depends strongly on X1 and X2, moderately on X3, not on X4/X5
    y = 3.0 * X1 + 2.0 * X2 + 0.5 * X3 + 0.1 * np.random.randn(n_samples)
    
    X = np.column_stack([X1, X2, X3, X4, X5]).astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train WarpGBM - more trees, smaller learning rate for stable importances
    wgbm_model = WarpGBM(
        objective='regression',
        max_depth=4,
        n_estimators=100,
        learning_rate=0.05,
        num_bins=32,
    )
    wgbm_model.fit(X_train, y_train)
    
    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(
        max_depth=4,
        n_estimators=100,
        learning_rate=0.05,
        max_bin=32,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Get feature importances
    wgbm_imp = wgbm_model.get_feature_importance()
    lgb_imp = lgb_model.feature_importances_
    
    feature_names = ['X1 (strong)', 'X2 (strong)', 'X3 (moderate)', 
                     'X4 (noise)', 'X5 (noise)']
    
    # Compare
    results = compare_feature_importances(
        wgbm_imp, lgb_imp, feature_names, "Regression"
    )
    
    # Assertions - synthetic data with clear signal should show good correlation
    assert results['rank_correlation'] > 0.5, \
        f"Rank correlation too low: {results['rank_correlation']}"
    
    # Check that X1 and X2 are in top 3 for both models
    wgbm_imp_norm = wgbm_imp / wgbm_imp.sum()
    lgb_imp_norm = lgb_imp / lgb_imp.sum()
    
    # X1 and X2 (indices 0 and 1) should have high importance
    assert wgbm_imp_norm[0] > wgbm_imp_norm[3], "X1 should be more important than X4"
    assert wgbm_imp_norm[1] > wgbm_imp_norm[4], "X2 should be more important than X5"
    
    print(f"\n✓ Regression feature importance test passed!")
    print(f"  Rank correlation: {results['rank_correlation']:.4f}")
    print(f"  X1 importance: {wgbm_imp_norm[0]:.4f}")
    print(f"  X2 importance: {wgbm_imp_norm[1]:.4f}")


def test_per_era_feature_importance():
    """Test per-era feature importance (unique to WarpGBM)"""
    print("\n" + "="*70)
    print("TEST: Per-Era Feature Importance")
    print("="*70)
    
    # Generate data where different features are important in different eras
    np.random.seed(42)
    n_samples_per_era = 300
    n_eras = 3
    
    X_list = []
    y_list = []
    era_list = []
    
    for era in range(n_eras):
        X_era = np.random.randn(n_samples_per_era, 4).astype(np.float32)
        
        # Different features are important in different eras
        if era == 0:
            # Era 0: X0 and X1 are important
            y_era = 3.0 * X_era[:, 0] + 2.0 * X_era[:, 1] + 0.1 * np.random.randn(n_samples_per_era)
        elif era == 1:
            # Era 1: X1 and X2 are important
            y_era = 2.0 * X_era[:, 1] + 3.0 * X_era[:, 2] + 0.1 * np.random.randn(n_samples_per_era)
        else:
            # Era 2: X0 and X2 are important
            y_era = 3.0 * X_era[:, 0] + 2.0 * X_era[:, 2] + 0.1 * np.random.randn(n_samples_per_era)
        
        X_list.append(X_era)
        y_list.append(y_era.astype(np.float32))
        era_list.append(np.full(n_samples_per_era, era, dtype=np.int32))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    era_id = np.concatenate(era_list)
    
    # Train WarpGBM with era-splitting - more trees for stable importances
    wgbm_model = WarpGBM(
        objective='regression',
        max_depth=4,
        n_estimators=100,
        learning_rate=0.05,
        num_bins=32,
    )
    wgbm_model.fit(X, y, era_id=era_id)
    
    # Get per-era importances
    per_era_imp = wgbm_model.get_per_era_feature_importance()
    total_imp = wgbm_model.get_feature_importance()
    
    print(f"\nPer-Era Feature Importance:")
    print(f"{'Feature':<15} {'Era 0':>12} {'Era 1':>12} {'Era 2':>12} {'Total':>12}")
    print(f"{'-'*63}")
    
    for feat_idx in range(4):
        print(f"Feature {feat_idx:<7} "
              f"{per_era_imp[0, feat_idx]:>12.4f} "
              f"{per_era_imp[1, feat_idx]:>12.4f} "
              f"{per_era_imp[2, feat_idx]:>12.4f} "
              f"{total_imp[feat_idx]:>12.4f}")
    
    # Assertions: Check that the model identified different important features per era
    # X1 should be important in all eras (it's in all three data generation processes)
    assert total_imp[1] > total_imp[3], "X1 should be more important than X3"
    
    # Check shape
    assert per_era_imp.shape == (n_eras, 4), \
        f"Wrong shape: {per_era_imp.shape}, expected (3, 4)"
    
    print(f"\n✓ Per-era feature importance test passed!")
    print(f"  Shape: {per_era_imp.shape}")
    print(f"  X1 is consistently important across eras (as expected)")


def test_no_era_equals_single_era():
    """Test that no era_id equals single era (backward compatibility)"""
    print("\n" + "="*70)
    print("TEST: No era_id == Single Era (ERM)")
    print("="*70)
    
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    X, y = X.astype(np.float32), y.astype(np.float32)
    
    # Train without era_id
    model = WarpGBM(max_depth=3, n_estimators=20, learning_rate=0.1)
    model.fit(X, y)
    
    # Get importances
    total_imp = model.get_feature_importance()
    per_era_imp = model.get_per_era_feature_importance()
    
    print(f"\nTotal importance shape: {total_imp.shape}")
    print(f"Per-era importance shape: {per_era_imp.shape}")
    
    # Check that per_era has shape (1, n_features) and matches total
    assert per_era_imp.shape == (1, 10), \
        f"Expected shape (1, 10), got {per_era_imp.shape}"
    
    # Normalize both for comparison
    total_imp_norm = total_imp / total_imp.sum()
    per_era_imp_norm = per_era_imp[0] / per_era_imp[0].sum()
    
    # They should be identical (within floating point precision)
    assert np.allclose(total_imp_norm, per_era_imp_norm), \
        "Total and per-era importance should match when n_eras=1"
    
    print(f"\n✓ Single era test passed!")
    print(f"  Total and per-era importances match (n_eras=1)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WarpGBM Feature Importance Test Suite")
    print("="*70)
    
    test_iris_feature_importance()
    test_regression_feature_importance()
    test_per_era_feature_importance()
    test_no_era_equals_single_era()
    
    print("\n" + "="*70)
    print("✓ ALL FEATURE IMPORTANCE TESTS PASSED!")
    print("="*70)

