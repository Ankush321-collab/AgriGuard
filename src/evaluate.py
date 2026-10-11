"""Evaluation visualization and reporting module for AgriGuard-ML."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrices(
    confusion_matrices: list[tuple[str, np.ndarray]],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot confusion matrices in a grid for all models and ensembles."""
    n_models = len(confusion_matrices)
    n_cols = 3 if n_models > 4 else 2
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices):
        ax = axes_flat[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, rotation=0, fontsize=8)

    # Hide any unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved confusion matrices figure to: {output_path}")


def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot model accuracy and macro-F1 comparison chart."""
    df_sorted = comparison_df.sort_values("Macro-F1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df_sorted) * 0.8)))
    y_positions = np.arange(len(df_sorted))
    bar_height = 0.35

    ax.barh(y_positions - bar_height / 2, df_sorted["Accuracy"], height=bar_height, label="Accuracy", color="#2a75d3")
    ax.barh(y_positions + bar_height / 2, df_sorted["Macro-F1"], height=bar_height, label="Macro-F1", color="#2f6f4e")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_sorted["Model"], fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title("AgriGuard-ML: Model & Ensemble Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (acc, f1) in enumerate(zip(df_sorted["Accuracy"], df_sorted["Macro-F1"])):
        ax.text(acc + 0.01, i - bar_height / 2, f"{acc:.3f}", va="center", fontsize=8, color="#1e4e8c")
        ax.text(f1 + 0.01, i + bar_height / 2, f"{f1:.3f}", va="center", fontsize=8, color="#1e4f35")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved metric comparison figure to: {output_path}")
