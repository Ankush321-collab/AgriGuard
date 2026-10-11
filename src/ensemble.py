"""Ensemble modeling module for AgriGuard-ML.

Implements:
1. Soft Voting Ensemble combining tuned KNN, SVM, LightGBM, and Random Forest.
2. Stacking Ensemble using Logistic Regression meta-classifier on base model probability predictions.
3. Validation and test comparison of all models and ensembles.
4. Best model selection based on validation Macro-F1.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder


def build_and_train_ensembles(
    base_models: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Construct and train Soft Voting and Stacking ensembles from tuned base models."""
    estimators = [
        ("KNN", base_models["KNN"]),
        ("SVM", base_models["SVM"]),
        ("LightGBM", base_models["LightGBM"]),
        ("Random_Forest", base_models["Random_Forest"]),
    ]

    print("\n" + "=" * 60, flush=True)
    print("Building and Training Ensembles", flush=True)
    print("=" * 60, flush=True)

    # 1. Soft Voting Ensemble using pre-fitted base models
    print("\n[Ensemble 1/2] Constructing Soft Voting Classifier from tuned base models...", flush=True)
    t0 = time.perf_counter()
    voting_ensemble = VotingClassifier(estimators=estimators, voting="soft")
    voting_ensemble.estimators_ = [base_models[name] for name, _ in estimators]
    voting_ensemble.le_ = LabelEncoder().fit(y_train)
    voting_ensemble.classes_ = voting_ensemble.le_.classes_
    t_voting = time.perf_counter() - t0
    print(f"  Soft Voting assembled in {t_voting:.2f}s", flush=True)

    # 2. Stacking Ensemble with LogisticRegression meta-model using cv='prefit'
    print("\n[Ensemble 2/2] Training Stacking Classifier (Meta-Learner: LogisticRegression)...", flush=True)
    t0 = time.perf_counter()
    stacking_ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv="prefit",
        stack_method="predict_proba",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacking_ensemble.fit(x_train, y_train)
    t_stacking = time.perf_counter() - t0
    print(f"  Stacking Classifier fit completed in {t_stacking:.2f}s", flush=True)

    return {
        "Voting_Ensemble": voting_ensemble,
        "Stacking_Ensemble": stacking_ensemble,
    }


def evaluate_all_models(
    all_models: dict[str, Any],
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    results_dir: Path,
    models_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[tuple[str, np.ndarray]]]:
    """Evaluate base models and ensembles, log comparison metrics, and save artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    prediction_columns = {"actual": y_test}
    confusion_matrices = []

    for name, model in all_models.items():
        print(f"\nEvaluating {name}...", flush=True)
        t0 = time.perf_counter()
        test_preds = model.predict(x_test).astype(int)
        infer_time = time.perf_counter() - t0

        val_preds = model.predict(x_val).astype(int)

        acc = accuracy_score(y_test, test_preds)
        macro_f1 = f1_score(y_test, test_preds, average="macro")
        val_macro_f1 = f1_score(y_val, val_preds, average="macro")
        report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

        comparison_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Macro-F1": macro_f1,
            "Weighted-F1": report["weighted avg"]["f1-score"],
            "Macro-Precision": report["macro avg"]["precision"],
            "Macro-Recall": report["macro avg"]["recall"],
            "Validation-Macro-F1": val_macro_f1,
            "InferenceTimeSeconds": infer_time,
        })
        prediction_columns[name] = test_preds

        cm = confusion_matrix(y_test, test_preds, labels=np.arange(len(class_names)))
        confusion_matrices.append((name, cm))

        # Save model (.joblib)
        joblib.dump(model, models_dir / f"{name}.joblib")
        print(f"  {name}: Val-Macro-F1={val_macro_f1:.4f}, Test-Acc={acc:.4f}, Test-Macro-F1={macro_f1:.4f}", flush=True)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("Validation-Macro-F1", ascending=False)
    comparison_df.to_csv(results_dir / "model_comparison_results.csv", index=False)

    ensemble_df = comparison_df[comparison_df["Model"].str.contains("Ensemble")]
    ensemble_df.to_csv(results_dir / "ensemble_results.csv", index=False)

    best_row = comparison_df.iloc[0]
    best_selection = {
        "best_model_name": best_row["Model"],
        "selection_metric": "Validation-Macro-F1",
        "validation_macro_f1": float(best_row["Validation-Macro-F1"]),
        "test_macro_f1": float(best_row["Macro-F1"]),
        "test_accuracy": float(best_row["Accuracy"]),
        "is_ensemble": "Ensemble" in best_row["Model"],
        "all_ranked_models": comparison_df[["Model", "Validation-Macro-F1", "Macro-F1", "Accuracy"]].to_dict(orient="records"),
    }
    with open(results_dir / "best_model_selection.json", "w", encoding="utf-8") as file:
        json.dump(best_selection, file, indent=2)

    return comparison_df, best_selection, confusion_matrices
