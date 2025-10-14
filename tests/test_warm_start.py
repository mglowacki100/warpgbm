"""
Tests for warm_start and model save/load functionality.
"""
import pytest
import numpy as np
import os
import tempfile
from sklearn.datasets import make_regression, make_classification, load_iris
from warpgbm import WarpGBM


class TestWarmStartRegression:
    """Test warm_start functionality for regression."""
    
    def test_warm_start_equivalent_to_full_training(self):
        """
        Train model with 100 trees at once vs 50 + 50 with warm_start.
        Predictions should be identical.
        """
        # Generate synthetic data
        X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
        X_test, y_test = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=43)
        
        # Model 1: Train 100 trees at once
        model_full = WarpGBM(
            objective="regression",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model_full.fit(X, y)
        preds_full = model_full.predict(X_test)
        
        # Model 2: Train 50 trees, then 50 more with warm_start
        model_incremental = WarpGBM(
            objective="regression",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            warm_start=True
        )
        model_incremental.fit(X, y)
        assert model_incremental._trees_trained == 50
        
        # Continue training for 50 more trees
        model_incremental.n_estimators = 100
        model_incremental.fit(X, y)
        assert model_incremental._trees_trained == 100
        
        preds_incremental = model_incremental.predict(X_test)
        
        # Predictions should be identical (or very close due to GPU non-determinism)
        np.testing.assert_allclose(preds_full, preds_incremental, rtol=1e-3, atol=1e-3)
        print(f"✓ Regression warm start: predictions match (max diff: {np.abs(preds_full - preds_incremental).max():.6f})")
    
    def test_warm_start_false_resets_model(self):
        """
        With warm_start=False (default), calling fit() should retrain from scratch.
        """
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        
        model = WarpGBM(
            objective="regression",
            n_estimators=30,
            max_depth=3,
            random_state=42,
            warm_start=False  # Explicit
        )
        model.fit(X, y)
        preds_1 = model.predict(X)
        
        # Fit again - should reset and retrain
        model.fit(X, y)
        preds_2 = model.predict(X)
        
        # Should be identical since same random_state and fresh training
        np.testing.assert_allclose(preds_1, preds_2, rtol=1e-3, atol=1e-3)
        assert model._trees_trained == 30
        print("✓ Regression warm_start=False: model resets correctly")
    
    def test_warm_start_already_complete(self):
        """
        If model already has n_estimators trees, calling fit() with warm_start should do nothing.
        """
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        model = WarpGBM(
            objective="regression",
            n_estimators=20,
            random_state=42,
            warm_start=True
        )
        model.fit(X, y)
        assert model._trees_trained == 20
        
        # Fit again with same n_estimators - should do nothing
        model.fit(X, y)
        assert model._trees_trained == 20
        print("✓ Regression warm start with complete model: no extra training")


class TestWarmStartClassification:
    """Test warm_start functionality for classification."""
    
    def test_multiclass_warm_start_equivalent(self):
        """
        Train multiclass model with 60 rounds at once vs 30 + 30 with warm_start.
        Predictions should be identical.
        """
        # Use Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split for testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Model 1: Train 60 rounds at once
        model_full = WarpGBM(
            objective="multiclass",
            n_estimators=60,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model_full.fit(X_train, y_train)
        preds_full = model_full.predict(X_test)
        probs_full = model_full.predict_proba(X_test)
        
        # Model 2: Train 30 rounds, then 30 more with warm_start
        model_incremental = WarpGBM(
            objective="multiclass",
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            warm_start=True
        )
        model_incremental.fit(X_train, y_train)
        assert model_incremental._trees_trained == 30
        
        # Continue training for 30 more rounds
        model_incremental.n_estimators = 60
        model_incremental.fit(X_train, y_train)
        assert model_incremental._trees_trained == 60
        
        preds_incremental = model_incremental.predict(X_test)
        probs_incremental = model_incremental.predict_proba(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(preds_full, preds_incremental)
        np.testing.assert_allclose(probs_full, probs_incremental, rtol=1e-3, atol=1e-3)
        print(f"✓ Multiclass warm start: predictions match (max prob diff: {np.abs(probs_full - probs_incremental).max():.6f})")
    
    def test_binary_warm_start(self):
        """Test warm_start with binary classification."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        # Full training
        model_full = WarpGBM(
            objective="binary",
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model_full.fit(X, y)
        preds_full = model_full.predict(X)
        
        # Incremental training
        model_incremental = WarpGBM(
            objective="binary",
            n_estimators=20,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            warm_start=True
        )
        model_incremental.fit(X, y)
        model_incremental.n_estimators = 40
        model_incremental.fit(X, y)
        preds_incremental = model_incremental.predict(X)
        
        # Predictions should match
        np.testing.assert_array_equal(preds_full, preds_incremental)
        print("✓ Binary classification warm start: predictions match")


class TestSaveLoad:
    """Test save_model() and load_model() functionality."""
    
    def test_save_load_regression_predictions_identical(self):
        """
        Save and load a regression model.
        Predictions before and after should be identical.
        """
        X, y = make_regression(n_samples=200, n_features=8, random_state=42)
        X_test, _ = make_regression(n_samples=50, n_features=8, random_state=43)
        
        # Train model
        model = WarpGBM(
            objective="regression",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        preds_before = model.predict(X_test)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Load model into new instance
            model_loaded = WarpGBM()
            model_loaded.load_model(tmp_path)
            preds_after = model_loaded.predict(X_test)
            
            # Predictions should be identical (allow tiny floating point differences)
            np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-5)
            assert model_loaded._trees_trained == 50
            assert model_loaded._is_fitted == True
            print("✓ Regression save/load: predictions identical")
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_load_multiclass_predictions_identical(self):
        """
        Save and load a multiclass model.
        Predictions and probabilities should be identical.
        """
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train model
        model = WarpGBM(
            objective="multiclass",
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        preds_before = model.predict(X)
        probs_before = model.predict_proba(X)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Load model
            model_loaded = WarpGBM()
            model_loaded.load_model(tmp_path)
            preds_after = model_loaded.predict(X)
            probs_after = model_loaded.predict_proba(X)
            
            # Predictions should be identical
            np.testing.assert_array_equal(preds_before, preds_after)
            np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5, atol=1e-5)
            assert model_loaded.num_classes == 3
            # classes_ stores the unique labels from y (0, 1, 2), not the string names
            np.testing.assert_array_equal(model_loaded.classes_, np.array([0, 1, 2]))
            print("✓ Multiclass save/load: predictions identical")
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_load_then_resume_training(self):
        """
        Train model, save it, load it, then continue training with warm_start.
        Final model should have more trees and work correctly.
        """
        X, y = make_regression(n_samples=300, n_features=10, random_state=42)
        X_test, _ = make_regression(n_samples=100, n_features=10, random_state=43)
        
        # Train 50 trees
        model = WarpGBM(
            objective="regression",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            warm_start=True
        )
        model.fit(X, y)
        assert model._trees_trained == 50
        preds_50 = model.predict(X_test)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Load model and continue training
            model_loaded = WarpGBM()
            model_loaded.load_model(tmp_path)
            model_loaded.warm_start = True  # Enable warm start
            model_loaded.n_estimators = 100  # Train 50 more trees
            model_loaded.fit(X, y)
            
            assert model_loaded._trees_trained == 100
            preds_100 = model_loaded.predict(X_test)
            
            # Predictions should be different (more trees = better fit usually)
            assert not np.array_equal(preds_50, preds_100)
            
            # Compare with model trained with 100 trees from scratch
            model_full = WarpGBM(
                objective="regression",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model_full.fit(X, y)
            preds_full = model_full.predict(X_test)
            
            # Should match the full training
            np.testing.assert_allclose(preds_100, preds_full, rtol=1e-3, atol=1e-3)
            print("✓ Save/load then resume training: works correctly")
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_load_multiclass_resume_training(self):
        """
        Save multiclass model, load it, and continue training.
        """
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=8,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42
        )
        
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train 30 rounds
        model = WarpGBM(
            objective="multiclass",
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            warm_start=True
        )
        model.fit(X_train, y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Load and continue training
            model_loaded = WarpGBM()
            model_loaded.load_model(tmp_path)
            model_loaded.warm_start = True
            model_loaded.n_estimators = 60  # Train 30 more rounds
            model_loaded.fit(X_train, y_train)
            
            assert model_loaded._trees_trained == 60
            
            # Compare with full training
            model_full = WarpGBM(
                objective="multiclass",
                n_estimators=60,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model_full.fit(X_train, y_train)
            
            preds_loaded = model_loaded.predict(X_test)
            preds_full = model_full.predict(X_test)
            
            np.testing.assert_array_equal(preds_loaded, preds_full)
            print("✓ Multiclass save/load/resume: works correctly")
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestFeatureImportanceWithWarmStart:
    """Test that feature importance is preserved and updated correctly with warm_start."""
    
    def test_feature_importance_accumulated(self):
        """
        Feature importance should accumulate when using warm_start.
        """
        X, y = make_regression(n_samples=300, n_features=10, random_state=42)
        
        # Train with warm start in two stages
        model = WarpGBM(
            objective="regression",
            n_estimators=30,
            max_depth=3,
            random_state=42,
            warm_start=True
        )
        model.fit(X, y)
        importance_30 = model.get_feature_importance(normalize=False)
        
        # Continue training
        model.n_estimators = 60
        model.fit(X, y)
        importance_60 = model.get_feature_importance(normalize=False)
        
        # Importance should increase (more trees contribute)
        assert np.all(importance_60 >= importance_30)
        print("✓ Feature importance accumulates correctly with warm_start")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

