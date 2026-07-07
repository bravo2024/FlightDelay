"""visualizations.py - Flight delay analysis plots."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#fafafa",
        "text.color": "#fafafa",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


def plot_delay_by_hour(hourly: Dict) -> plt.Figure:
    """Line chart of delay rate by departure hour."""
    _style()
    hours = sorted(hourly.keys())
    rates = [hourly[h]["delay_rate"] for h in hours]
    counts = [hourly[h]["count"] for h in hours]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    line1 = ax1.plot(hours, rates, "o-", color="#f43f5e", linewidth=2, markersize=5, label="Delay Rate")
    bars = ax2.bar(hours, counts, alpha=0.15, color="#22d3ee", label="Flight Count")

    ax1.set_xlabel("Departure Hour")
    ax1.set_ylabel("Delay Rate (%)", color="#f43f5e")
    ax2.set_ylabel("Number of Flights", color="#22d3ee")

    # Mark rush hours
    for h in [7, 8, 9, 17, 18, 19]:
        ax1.axvspan(h - 0.4, h + 0.4, alpha=0.1, color="#f97316")

    ax1.set_title("Delay Rate by Departure Hour", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(range(24))
    ax1.grid(True, linestyle="--")

    # Combined legend
    lines = line1 + [bars]
    labels = [l.get_label() for l in line1] + ["Flight Count"]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


def plot_delay_by_month(monthly_data: Dict) -> plt.Figure:
    """Bar chart of delay rate by month."""
    _style()
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rates = [monthly_data.get(m, {}).get("delay_rate", 0) for m in months]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#f43f5e" if r > 0.25 else "#fbbf24" if r > 0.15 else "#22c55e" for r in rates]
    bars = ax.bar(month_names, rates, color=colors, width=0.6, edgecolor="#333")

    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{rate:.1%}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Delay Rate")
    ax.set_title("Delay Rate by Month", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "") -> plt.Figure:
    """ROC curve with AUC annotation."""
    _style()
    # Compute ROC curve points
    thresholds = np.sort(np.unique(y_proba))[::-1]
    tpr_list = [0.0]
    fpr_list = [0.0]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # AUC via trapezoidal rule
    auc = np.trapz(tpr_list, fpr_list)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_list, tpr_list, color="#22d3ee", linewidth=2, label=f'{model_name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1, label="Random")
    ax.fill_between(fpr_list, tpr_list, alpha=0.1, color="#22d3ee")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray, std: np.ndarray,
                            feature_names: list, top_n: int = 14) -> plt.Figure:
    """Horizontal bar chart of permutation feature importance."""
    _style()
    n = min(top_n, len(feature_names))
    sorted_idx = np.argsort(importances)[-n:]
    names = [feature_names[i] for i in sorted_idx]
    vals = importances[sorted_idx]
    errs = std[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#f97316" if v > 0 else "#22c55e" for v in vals]
    ax.barh(names, vals, xerr=errs, color=colors, height=0.6, edgecolor="#333", capsize=3)
    ax.set_xlabel("Importance (Δ Accuracy)")
    ax.set_title("Feature Importance (Permutation)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--")
    ax.axvline(x=0, color="#666", linewidth=1)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """Annotated confusion matrix."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            ax.text(j, i, f"{val:,}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val > cm.max() / 2 else "#333")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["On-Time", "Delayed"], fontsize=10)
    ax.set_yticklabels(["On-Time", "Delayed"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_delay_distribution(y_true: np.ndarray) -> plt.Figure:
    """Bar chart of class distribution."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = [int((y_true == 0).sum()), int((y_true == 1).sum())]
    colors = ["#22c55e", "#f43f5e"]
    bars = ax.bar(["On-Time", "Delayed"], counts, color=colors, width=0.5, edgecolor="#333")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution (FAA ≥15 min threshold)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict: Dict) -> plt.Figure:
    """Grouped bar chart comparing models."""
    _style()
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    x = np.arange(len(metric_names))
    width = 0.3
    colors = ["#22d3ee", "#a78bfa", "#f97316"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(model_names):
        vals = [metrics_dict[name].get(m, 0) for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)],
                      edgecolor="#333")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names], fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    return fig


def plot_cost_analysis(thresholds: np.ndarray, fpr: np.ndarray, fnr: np.ndarray,
                       cost_fn: float = 500, cost_fp: float = 100) -> plt.Figure:
    """Business cost vs threshold."""
    _style()
    total_cost = fnr * cost_fn + fpr * cost_fp
    best_idx = np.argmin(total_cost)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, total_cost, color="#f43f5e", linewidth=2, label="Total Cost")
    ax.plot(thresholds, fnr * cost_fn, "--", color="#f97316", alpha=0.7, label="Missed Delay Cost")
    ax.plot(thresholds, fpr * cost_fp, "--", color="#22d3ee", alpha=0.7, label="False Alarm Cost")
    ax.axvline(x=thresholds[best_idx], color="#22c55e", linestyle="-", linewidth=2,
               label=f"Optimal τ = {thresholds[best_idx]:.2f}")
    ax.scatter([thresholds[best_idx]], [total_cost[best_idx]], color="#22c55e", s=100, zorder=5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Business Cost Analysis", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--")
    fig.tight_layout()
    return fig
