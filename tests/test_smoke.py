"""tests/test_smoke.py - Smoke tests for FlightDelay pipeline."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_make_synthetic():
    from src.data import make_synthetic
    data = make_synthetic(n=500)
    assert data["X"].shape[0] == 500
    assert data["y"].shape[0] == 500
    assert 0.05 < data["delay_rate"] < 0.5


def test_cyclical_encoding():
    from src.data import make_synthetic
    data = make_synthetic(n=100)
    features = data["features"]
    sin_idx = features.index("dep_hour_sin")
    cos_idx = features.index("dep_hour_cos")
    # sin^2 + cos^2 should be close to 1
    s2c2 = data["X"][:, sin_idx] ** 2 + data["X"][:, cos_idx] ** 2
    assert np.allclose(s2c2, 1.0, atol=1e-6)


def test_train_gb():
    from src.data import make_synthetic
    from src.model import train_gradient_boosting, evaluate
    from src.core import temporal_split, StandardScaler
    data = make_synthetic(n=500)
    X_train, X_test, y_train, y_test = temporal_split(data["X"], data["y"])
    scaler = StandardScaler().fit(X_train)
    result = train_gradient_boosting(scaler.transform(X_train), y_train, n_estimators=20)
    eval_result = evaluate(result, scaler.transform(X_test), y_test)
    assert 0 < eval_result["metrics"]["accuracy"] <= 1
    assert 0 < eval_result["metrics"]["roc_auc"] <= 1


def test_permutation_importance():
    from src.data import make_synthetic
    from src.model import train_gradient_boosting, permutation_importance
    from src.core import temporal_split, StandardScaler
    data = make_synthetic(n=300)
    X_train, X_test, y_train, y_test = temporal_split(data["X"], data["y"])
    scaler = StandardScaler().fit(X_train)
    result = train_gradient_boosting(scaler.transform(X_train), y_train, n_estimators=10)
    imp = permutation_importance(result, scaler.transform(X_test), y_test,
                                 data["features"], n_repeats=2)
    assert imp["importances"].shape == (len(data["features"]),)


if __name__ == "__main__":
    test_make_synthetic()
    test_cyclical_encoding()
    test_train_gb()
    test_permutation_importance()
    print("All smoke tests passed!")
