import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from warpgbm import WarpGBM
import time


def test_multiclass_3_classes():
    """Test multiclass classification with 3 classes"""
    print("\n" + "="*70)
    print("TEST: 3-Class Classification")
    print("="*70)
    
    # Generate synthetic 3-class dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42,
        flip_y=0.1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Train model
    print(f"\nTraining on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    model = WarpGBM(
        objective="multiclass",
        max_depth=5,
        n_estimators=50,
        learning_rate=0.1,
        num_bins=32,
        min_child_weight=10,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"\n✓ Training completed in {train_time:.2f}s")
    print(f"  Number of classes: {model.num_classes}")
    print(f"  Classes: {model.classes_}")
    print(f"  Trees per round: {model.num_classes}")
    print(f"  Total rounds: {len(model.forest)}")
    
    # Predictions
    start = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    pred_time = time.time() - start
    
    print(f"\n✓ Prediction completed in {pred_time:.2f}s")
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test LogLoss:  {logloss:.4f}")
    
    # Validate predictions
    assert y_pred_proba.shape == (len(X_test), 3), "Wrong probability shape"
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"
    assert accuracy > 0.70, f"Accuracy too low: {accuracy}"
    assert logloss < 1.0, f"LogLoss too high: {logloss}"
    
    print("\n✓ All assertions passed!")
    return model


def test_multiclass_blobs():
    """Test with well-separated blob data (should get near-perfect accuracy)"""
    print("\n" + "="*70)
    print("TEST: Well-Separated Blobs (5 classes)")
    print("="*70)
    
    # Generate well-separated blobs
    X, y = make_blobs(
        n_samples=2000,
        n_features=10,
        centers=5,
        cluster_std=1.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"\nTraining on {X_train.shape[0]} samples")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    model = WarpGBM(
        objective="multiclass",
        max_depth=4,
        n_estimators=30,
        learning_rate=0.3,
        num_bins=16,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"\n✓ Training completed in {train_time:.2f}s")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test LogLoss:  {logloss:.4f}")
    
    # Well-separated data should get very high accuracy
    assert accuracy > 0.95, f"Accuracy too low for well-separated data: {accuracy}"
    
    print("\n✓ All assertions passed!")
    return model


def test_binary_classification():
    """Test binary classification (special case of multiclass)"""
    print("\n" + "="*70)
    print("TEST: Binary Classification")
    print("="*70)
    
    X, y = make_classification(
        n_samples=3000,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"\nTraining on {X_train.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    model = WarpGBM(
        objective="binary",
        max_depth=5,
        n_estimators=40,
        learning_rate=0.1,
        num_bins=32,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"\n✓ Training completed in {train_time:.2f}s")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.75, f"Accuracy too low: {accuracy}"
    assert y_pred_proba.shape[1] == 2, "Binary classification should have 2 probability columns"
    
    print("\n✓ All assertions passed!")
    return model


def test_multiclass_with_eval():
    """Test multiclass with evaluation set and early stopping"""
    print("\n" + "="*70)
    print("TEST: Multiclass with Eval Set & Early Stopping")
    print("="*70)
    
    X, y = make_classification(
        n_samples=4000,
        n_features=20,
        n_informative=15,
        n_classes=4,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    model = WarpGBM(
        objective="multiclass",
        max_depth=5,
        n_estimators=200,  # More than needed, will early stop
        learning_rate=0.05,  # Slower learning to test early stopping
        num_bins=32,
    )
    
    start = time.time()
    model.fit(
        X_train, y_train,
        X_eval=X_val, y_eval=y_val,
        eval_every_n_trees=10,
        early_stopping_rounds=5,  # Stop if no improvement for 5 evals
        eval_metric="logloss"
    )
    train_time = time.time() - start
    
    print(f"\n✓ Training completed in {train_time:.2f}s")
    print(f"  Stopped at round: {len(model.forest)}")
    print(f"  Training loss history length: {len(model.training_loss)}")
    print(f"  Eval loss history length: {len(model.eval_loss)}")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.65, f"Accuracy too low: {accuracy}"
    # Check that evaluation was performed
    assert len(model.eval_loss) > 0, "Eval loss not recorded"
    print(f"  Early stopping {'triggered' if len(model.forest) < 200 else 'did not trigger'}")
    
    print("\n✓ All assertions passed!")
    return model


def test_string_labels():
    """Test that string labels work correctly"""
    print("\n" + "="*70)
    print("TEST: String Class Labels")
    print("="*70)
    
    X, y_numeric = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to string labels
    label_map = {0: "cat", 1: "dog", 2: "bird"}
    y = np.array([label_map[i] for i in y_numeric])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"\nClasses: {np.unique(y_train)}")
    
    model = WarpGBM(
        objective="multiclass",
        max_depth=4,
        n_estimators=20,
        learning_rate=0.2,
        num_bins=16,
    )
    
    model.fit(X_train, y_train)
    
    print(f"\nEncoded classes: {model.classes_}")
    
    y_pred = model.predict(X_test)
    
    # Check that predictions are strings
    assert isinstance(y_pred[0], str) or isinstance(y_pred[0], np.str_), \
        "Predictions should be strings"
    assert set(y_pred).issubset(set(["cat", "dog", "bird"])), \
        "Invalid predicted labels"
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.60, f"Accuracy too low: {accuracy}"
    
    print("\n✓ All assertions passed!")
    return model


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WarpGBM Multiclass Classification Test Suite")
    print("="*70)
    
    test_multiclass_3_classes()
    test_multiclass_blobs()
    test_binary_classification()
    test_multiclass_with_eval()
    test_string_labels()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)

