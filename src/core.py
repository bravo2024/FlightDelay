"""core.py - ML primitives for flight delay prediction (pure NumPy + sklearn fallback)."""
from __future__ import annotations
import numpy as np
import warnings


def temporal_split(X, y, test_size=0.2):
    """Chronological train/test split for time-series data.
    Preserves temporal order — the first (1-test_size)*100 % of rows
    form the training set.  For flight delay data this ensures that
    no future flight patterns leak into the past."""
    X, y = np.asarray(X, float), np.asarray(y)
    sp = max(1, int(len(X) * (1 - test_size)))
    return X[:sp], X[sp:], y[:sp], y[sp:]


def stratified_split(X, y, test_size=0.2, seed=42):
    """Stratified train/test split preserving class proportions.

    WARNING: This function *shuffles* data within each class, which
    destroys temporal order.  For flight delay prediction, use
    temporal_split() instead.  This function is retained only for
    cases where temporal ordering is irrelevant (e.g. model-free
    calibration checks)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        n_test = max(1, int(len(class_idx) * test_size))
        test_idx.extend(class_idx[:n_test])
        train_idx.extend(class_idx[n_test:])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class Standardizer:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0, ddof=1) + 1e-8
        return self
    def transform(self, X):
        return (np.asarray(X, float) - self.mu_) / self.sd_
    def fit_transform(self, X):
        return self.fit(X).transform(X)


StandardScaler = Standardizer


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def log_loss(y_true, y_proba, eps=1e-12):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_proba, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true, y_proba):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_proba, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def roc_auc_score(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    npos = (y == 1).sum()
    nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def accuracy_score(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def precision_score(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0


def confusion_matrix(y_true, y_pred):
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
        "positive_rate": float(np.mean(y_true)),
    }


class LogisticRegression:
    """Binary logistic regression via gradient descent with L2 regularization.

    Model: P(y=1|x) = sigmoid(w^T x + b)
    Loss:  J(w) = -(1/n) * sum[y*log(p) + (1-y)*log(1-p)] + (lambda/2)*||w||^2
    Gradient: dJ/dw = (1/n) * X^T(p - y) + lambda*w
    """
    def __init__(self, lr=0.1, epochs=500, l2=1e-3, class_weight=True, seed=42):
        self.lr, self.epochs, self.l2, self.class_weight, self.seed = lr, epochs, l2, class_weight, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        if self.class_weight:
            pos = max(y.sum(), 1.0)
            neg = max((1 - y).sum(), 1.0)
            sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        else:
            sw = np.ones(n)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w_ + self.b_)
            err = (p - y) * sw
            self.w_ -= self.lr * (X.T @ err / n + self.l2 * self.w_)
            self.b_ -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, float) @ self.w_ + self.b_)

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)


class GradientBoostedTrees:
    """Binary gradient-boosted decision tree classifier (NumPy-only fallback).

    Implements Friedman (2001) 'Greedy Function Approximation' using
    regression stumps (max_depth=1) with logistic loss.

    F_m(x) = F_{m-1}(x) + eta * h_m(x)

    where h_m fits the negative gradient (pseudo-residuals) of the
    binomial log-likelihood loss.

    This is an intentionally simplified implementation for use when
    neither LightGBM nor scikit-learn is available.  Production
    deployments should use LightGBM."""
    def __init__(self, n_trees=200, max_depth=3, learning_rate=0.1, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)
        self._trees = []
        log_odds = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
        F = np.full(n, log_odds)
        for _ in range(self.n_trees):
            p = 1.0 / (1.0 + np.exp(-F))
            residuals = y - p
            tree = self._fit_stump(X, residuals)
            leaf_values = tree["leaf_values"]
            for i in range(n):
                F[i] += self.learning_rate * leaf_values[tree["assignment"][i]]
            self._trees.append(tree)
        return self

    def _fit_stump(self, X, residuals):
        n, d = X.shape
        if self.max_depth <= 1:
            col, thresh = self._best_split(X, residuals)
            left = X[:, col] <= thresh
            right = ~left
            leaf_values = np.zeros(2)
            for idx, mask in enumerate([left, right]):
                if mask.sum() > 0:
                    leaf_values[idx] = residuals[mask].mean()
            assignment = np.where(left, 0, 1)
            return {"col": col, "thresh": thresh, "leaf_values": leaf_values, "assignment": assignment}
        left = X[:, 0] <= X[:, 0].mean()
        lv = residuals[left].mean() if left.sum() > 0 else residuals.mean()
        rv = residuals[~left].mean() if (~left).sum() > 0 else residuals.mean()
        return {"col": 0, "thresh": X[:, 0].mean(),
                "leaf_values": np.array([lv, rv]),
                "assignment": np.where(left, 0, 1)}

    def _best_split(self, X, residuals):
        best_gain = -1.0
        best_col, best_thresh = 0, 0.0
        for col in range(X.shape[1]):
            sorted_idx = np.argsort(X[:, col])
            sorted_x = X[sorted_idx, col]
            sorted_r = residuals[sorted_idx]
            n_total = len(residuals)
            sum_total = residuals.sum()
            sum_left = 0.0
            count_left = 0
            for i in range(1, n_total):
                sum_left += sorted_r[i - 1]
                count_left += 1
                if sorted_x[i] == sorted_x[i - 1]:
                    continue
                sum_right = sum_total - sum_left
                count_right = n_total - count_left
                if count_left < 1 or count_right < 1:
                    continue
                gain = (sum_left ** 2) / count_left + (sum_right ** 2) / count_right
                if gain > best_gain:
                    best_gain = gain
                    best_col = col
                    best_thresh = (sorted_x[i - 1] + sorted_x[i]) / 2.0
        if best_gain < 0:
            return 0, X[:, 0].mean()
        return best_col, best_thresh

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        F = np.zeros(n)
        for tree in self._trees:
            col, thresh = tree["col"], tree["thresh"]
            left = X[:, col] <= thresh
            assignment = np.where(left, 0, 1)
            F += self.learning_rate * tree["leaf_values"][assignment]
        return 1.0 / (1.0 + np.exp(-F))

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)
