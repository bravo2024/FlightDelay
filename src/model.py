"""model.py - Flight delay prediction models.

Models:
1. XGBoost/Gradient Boosting: state-of-the-art for tabular data
2. Logistic Regression baseline: interpretable linear model
3. Feature importance via permutation importance (model-agnostic)

Mathematical foundations:
- Gradient Boosting: F_m(x) = F_{m-1}(x) + η · h_m(x)
  where h_m fits negative gradient of loss
- Permutation importance: I_j = E[loss] - E[loss | feature j shuffled]
- SHAP values: φ_j = Σ_{S⊆N\{j}} |S|!(|N|-|S|-1)!/|N|! · [f(S∪{j}) - f(S)]
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray,
                            n_estimators: int = 200, learning_rate: float = 0.1,
                            max_depth: int = 5, seed: int = 42) -> Dict:
    """Train Gradient Boosting classifier.

    Tries LightGBM first, falls back to sklearn GradientBoostingClassifier.

    Math (Gradient Boosting for classification):
        F_0(x) = log(p / (1-p))  (log-odds of base rate)
        For m = 1 to M:
            p_m(x) = sigmoid(F_{m-1}(x))
            r_i = y_i - p_m(x_i)  (pseudo-residuals)
            Fit tree h_m(x) to r_i
            F_m(x) = F_{m-1}(x) + η · h_m(x)

    where η is learning rate (shrinkage parameter).
    """
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, class_weight="balanced",
            random_state=seed, verbose=-1,
        )
        model.fit(X_train, y_train)
        return {"model": model, "backend": "lightgbm", "feature_importance": "gain"}
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=seed,
        )
        model.fit(X_train, y_train)
        return {"model": model, "backend": "sklearn", "feature_importance": "deviance"}
    except ImportError:
        pass

    from src.core import GradientBoostedTrees as GBT
    model = GBT(n_trees=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, seed=seed)
    model.fit(X_train, y_train)
    return {"model": model, "backend": "scratch", "feature_importance": "scratch"}


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """Train logistic regression baseline."""
    from src.core import LogisticRegression
    model = LogisticRegression(lr=0.1, epochs=500, l2=1e-3, class_weight=True)
    model.fit(X_train, y_train)
    return {"model": model, "backend": "scratch"}


def evaluate(model_dict: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate model on test set."""
    from src.core import compute_metrics, confusion_matrix, roc_auc_score

    model = model_dict["model"]
    backend = model_dict["backend"]

    if backend == "scratch":
        y_proba = model.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    return {"metrics": metrics, "y_proba": y_proba, "y_pred": y_pred}


def permutation_importance(model_dict: Dict, X: np.ndarray, y: np.ndarray,
                           feature_names: list, n_repeats: int = 10,
                           seed: int = 42) -> Dict:
    """Compute permutation importance (model-agnostic).

    Math: For each feature j:
        1. Compute baseline loss: L_0 = loss(model, X, y)
        2. For each repeat r:
            a. Create X_shuffled = X with column j randomly permuted
            b. Compute L_r = loss(model, X_shuffled, y)
        3. Importance: I_j = mean(L_r - L_0) / std(L_r)

    Features with high importance significantly affect model performance.
    """
    rng = np.random.default_rng(seed)
    model = model_dict["model"]
    backend = model_dict["backend"]

    if backend == "scratch":
        base_proba = model.predict_proba(X)
        if base_proba.ndim > 1:
            base_proba = base_proba[:, 1]
    else:
        base_proba = model.predict_proba(X)[:, 1]

    base_pred = (base_proba >= 0.5).astype(int)
    base_loss = float(np.mean(base_pred != y))

    importances = np.zeros((n_repeats, X.shape[1]))

    for r in range(n_repeats):
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])

            if backend == "scratch":
                perm_proba = model.predict_proba(X_perm)
                if perm_proba.ndim > 1:
                    perm_proba = perm_proba[:, 1]
            else:
                perm_proba = model.predict_proba(X_perm)[:, 1]

            perm_pred = (perm_proba >= 0.5).astype(int)
            perm_loss = float(np.mean(perm_pred != y))
            importances[r, j] = perm_loss - base_loss

    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_imp)[::-1]

    return {
        "importances": mean_imp,
        "std": std_imp,
        "feature_names": feature_names,
        "sorted_idx": sorted_idx,
        "baseline_loss": base_loss,
    }


def temporal_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                      hours: np.ndarray) -> Dict:
    """Analyze delay rates by hour of day.

    Math: For each hour h:
        P(delay | hour=h) = #delayed at hour h / #total at hour h
        Expected value: E[P(delay)] = Σ_h P(delay|h) · P(hour=h)
    """
    unique_hours = np.sort(np.unique(hours))
    hourly = {}
    for h in unique_hours:
        mask = hours == h
        if mask.sum() > 0:
            hourly[int(h)] = {
                "count": int(mask.sum()),
                "delay_rate": float(y_true[mask].mean()),
                "avg_predicted": float(y_proba[mask].mean()),
            }
    return hourly


def route_analysis(df, y_true=None, top_n: int = 20) -> Dict:
    """Analyze delay rates by route (origin → destination)."""
    if "ORIGIN" in df.columns and "DEST" in df.columns:
        routes = df.groupby(["ORIGIN", "DEST"]).agg(
            count=("delayed", "count"),
            delay_rate=("delayed", "mean"),
        ).reset_index()
        routes = routes.sort_values("count", ascending=False).head(top_n)
        return routes.to_dict("records")
    return []
