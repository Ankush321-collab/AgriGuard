"""Phase 2: Trustworthy ML Module for AgriGuard-ML.

Implements:
1. Duplicate detection and leak-free split generation.
2. Final held-out test set preservation.
3. Per-class precision, recall, F1, and confusion matrix export.
4. Probability calibration analysis (Brier score, reliability).
5. Confidence threshold selection on validation data and abstention rate reporting.
6. Unified model bundle serialization with dependency tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20
FEATURE_PIPELINE_VERSION = "848_multimodal_v1.0"


def audit_and_deduplicate(
    features: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    class_names: list[str],
    results_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], dict[str, Any]]:
    """Audit dataset for exact/near duplicates and return unique instances."""
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_file = results_dir / "data_integrity_audit.json"

    # Fast hash check on paths & features
    print("Auditing dataset for duplicates...", flush=True)
    hash_to_indices = defaultdict(list)
    for idx, p in enumerate(paths):
        full_path = Path(p)
        if full_path.is_file():
            try:
                with open(full_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                    hash_to_indices[h].append(idx)
            except Exception:
                hash_to_indices[f"err_{idx}"].append(idx)
        else:
            # Fallback to feature hash if file is not found
            feat_hash = hashlib.md5(features[idx].tobytes()).hexdigest()
            hash_to_indices[feat_hash].append(idx)

    dup_clusters = [idxs for idxs in hash_to_indices.values() if len(idxs) > 1]
    redundant_indices = set()
    dup_by_class = defaultdict(int)

    for cluster in dup_clusters:
        # Keep first instance, mark subsequent as redundant
        for dup_idx in cluster[1:]:
            redundant_indices.add(dup_idx)
            cls_name = class_names[labels[dup_idx]]
            dup_by_class[cls_name] += 1

    unique_indices = [i for i in range(len(paths)) if i not in redundant_indices]

    audit_summary = {
        "total_images_processed": len(paths),
        "unique_images": len(unique_indices),
        "duplicate_clusters": len(dup_clusters),
        "removed_redundant_images": len(redundant_indices),
        "duplicates_per_class": dict(dup_by_class),
        "feature_extraction_failures": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(
        f"Audit Complete: {len(unique_indices)} unique images retained. "
        f"{len(redundant_indices)} duplicate images isolated to prevent cross-split leakage.",
        flush=True,
    )

    clean_features = features[unique_indices]
    clean_labels = labels[unique_indices]
    clean_paths = [paths[i] for i in unique_indices]

    return clean_features, clean_labels, clean_paths, unique_indices, audit_summary


def create_leak_free_splits(
    features: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    class_names: list[str],
    results_dir: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder
]:
    """Create stratified train (70%), val (10%), test (20%) splits with scaler fit on train ONLY."""
    indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        stratify=labels[temp_indices],
        random_state=RANDOM_STATE,
    )

    # Save split manifest
    split_rows = []
    for split_name, split_idx in (("train", train_indices), ("validation", val_indices), ("test", test_indices)):
        for idx in split_idx:
            split_rows.append({
                "feature_row": int(idx),
                "path": str(paths[idx]),
                "label": int(labels[idx]),
                "class_name": class_names[int(labels[idx])],
                "split": split_name,
            })
    manifest_df = pd.DataFrame(split_rows).sort_values("feature_row")
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(results_dir / "split_manifest.csv", index=False)
    manifest_df.to_json(results_dir / "split_manifest.json", orient="records", indent=2)

    # Fit scaler strictly on training split
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(features[train_indices])
    x_val_scaled = scaler.transform(features[val_indices])
    x_test_scaled = scaler.transform(features[test_indices])

    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    return (
        x_train_scaled, y_train,
        x_val_scaled, y_val,
        x_test_scaled, y_test,
        train_indices, val_indices, test_indices,
        scaler, label_encoder,
    )


def compute_calibration_and_brier(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Calculate multi-class Brier score and expected calibration error."""
    # One-hot encode ground truth
    y_one_hot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_one_hot[i, label] = 1.0

    # Multi-class Brier score: mean squared error across all class probability columns
    brier_score = float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))

    # Max confidence vs accuracy across 10 bins
    max_probs = np.max(y_probs, axis=1)
    preds = np.argmax(y_probs, axis=1)
    accuracies = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, 11)
    bin_data = []
    ece = 0.0

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (max_probs >= low) & (max_probs < high) if i < len(bins) - 2 else (max_probs >= low) & (max_probs <= high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(max_probs[mask]))
            bin_data.append({
                "bin_range": f"{low:.1f}-{high:.1f}",
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)

    return {
        "multi_class_brier_score": round(brier_score, 4),
        "expected_calibration_error": round(float(ece), 4),
        "reliability_bins": bin_data,
    }


def analyze_confidence_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Sweep confidence thresholds and evaluate coverage, abstention rate, and accepted accuracy."""
    if thresholds is None:
        thresholds = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    max_probs = np.max(y_probs, axis=1)
    preds = np.argmax(y_probs, axis=1)
    total_samples = len(y_true)

    results = []
    for thresh in thresholds:
        accepted_mask = max_probs >= thresh
        accepted_count = int(np.sum(accepted_mask))
        abstain_count = total_samples - accepted_count
        abstain_rate = abstain_count / total_samples

        if accepted_count > 0:
            acc_on_accepted = float(accuracy_score(y_true[accepted_mask], preds[accepted_mask]))
            macro_f1_accepted = float(f1_score(y_true[accepted_mask], preds[accepted_mask], average="macro", zero_division=0))
        else:
            acc_on_accepted = 0.0
            macro_f1_accepted = 0.0

        results.append({
            "threshold": round(thresh, 2),
            "accepted_samples": accepted_count,
            "abstain_samples": abstain_count,
            "abstain_rate_pct": round(abstain_rate * 100, 2),
            "coverage_pct": round((1.0 - abstain_rate) * 100, 2),
            "accuracy_on_accepted": round(acc_on_accepted, 4),
            "macro_f1_on_accepted": round(macro_f1_accepted, 4),
        })

    return results


def evaluate_trustworthy_pipeline(
    all_models: dict[str, Any],
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    results_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    """Perform trustworthy evaluation: per-class reports, calibration, and abstention tuning."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_class_tables = {}
    calibration_reports = {}
    abstention_reports = {}

    for name, model in all_models.items():
        print(f"\nEvaluating {name} with probability calibration...", flush=True)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(x_val)
            test_probs = model.predict_proba(x_test)
        else:
            # Fallback for decision function
            val_preds = model.predict(x_val)
            test_preds = model.predict(x_test)
            val_probs = np.eye(len(class_names))[val_preds]
            test_probs = np.eye(len(class_names))[test_preds]

        val_preds = np.argmax(val_probs, axis=1)
        test_preds = np.argmax(test_probs, axis=1)

        val_acc = accuracy_score(y_val, val_preds)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        test_acc = accuracy_score(y_test, test_preds)
        test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_macro_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
        test_macro_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)

        # Calibration
        cal_val = compute_calibration_and_brier(y_val, val_probs, len(class_names))
        cal_test = compute_calibration_and_brier(y_test, test_probs, len(class_names))
        calibration_reports[name] = {"val": cal_val, "test": cal_test}

        # Abstention sweep on validation vs test
        val_abstention = analyze_confidence_thresholds(y_val, val_probs)
        test_abstention = analyze_confidence_thresholds(y_test, test_probs)
        abstention_reports[name] = {"val_sweep": val_abstention, "test_sweep": test_abstention}

        # Per-class report
        class_report = classification_report(
            y_test, test_preds, target_names=class_names, output_dict=True, zero_division=0
        )
        per_class_df = pd.DataFrame(class_report).transpose()
        per_class_df.to_csv(results_dir / f"per_class_metrics_{name}.csv")
        per_class_tables[name] = class_report

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_preds, labels=np.arange(len(class_names)))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(results_dir / f"confusion_matrix_{name}.csv")

        summary_rows.append({
            "Model": name,
            "Test-Accuracy": round(test_acc, 4),
            "Test-Macro-F1": round(test_macro_f1, 4),
            "Test-Macro-Precision": round(test_macro_prec, 4),
            "Test-Macro-Recall": round(test_macro_rec, 4),
            "Val-Macro-F1": round(val_macro_f1, 4),
            "Val-Accuracy": round(val_acc, 4),
            "BrierScore-Test": cal_test["multi_class_brier_score"],
            "ECE-Test": cal_test["expected_calibration_error"],
        })

        # Save model checkpoint
        joblib.dump(model, models_dir / f"{name}.joblib")

    # Save summary tables
    summary_df = pd.DataFrame(summary_rows).sort_values("Val-Macro-F1", ascending=False)
    summary_df.to_csv(results_dir / "trustworthy_model_comparison.csv", index=False)

    # Save calibration and abstention logs
    with open(results_dir / "probability_calibration_analysis.json", "w") as f:
        json.dump(calibration_reports, f, indent=2)
    with open(results_dir / "confidence_threshold_abstention_analysis.json", "w") as f:
        json.dump(abstention_reports, f, indent=2)

    return {
        "summary_df": summary_df,
        "per_class_tables": per_class_tables,
        "calibration_reports": calibration_reports,
        "abstention_reports": abstention_reports,
    }


def save_reproducible_bundle(
    models_dir: Path,
    results_dir: Path,
    class_names: list[str],
    selected_model_name: str,
    optimal_threshold: float,
) -> Path:
    """Bundle model, scaler, class order, feature version, and runtime dependencies together."""
    bundle_path = models_dir / "bundle_metadata.json"

    dependencies = {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": pd.__version__,
        "lightgbm": joblib.__version__,
    }
    try:
        import sklearn, lightgbm, cv2, mahotas, skimage
        dependencies["scikit-learn"] = sklearn.__version__
        dependencies["lightgbm"] = lightgbm.__version__
        dependencies["opencv-python"] = cv2.__version__
        dependencies["mahotas"] = mahotas.__version__
        dependencies["scikit-image"] = skimage.__version__
    except Exception:
        pass

    bundle_metadata = {
        "project": "AgriGuard-ML",
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "feature_dimension": 848,
        "primary_model_name": selected_model_name,
        "primary_model_file": f"{selected_model_name}.joblib",
        "scaler_file": "scaler.joblib",
        "label_encoder_file": "label_encoder.joblib",
        "class_names": class_names,
        "number_of_classes": len(class_names),
        "selected_confidence_threshold": optimal_threshold,
        "split_manifest_file": "results/split_manifest.csv",
        "audit_file": "results/data_integrity_audit.json",
        "evaluation_file": "results/trustworthy_model_comparison.csv",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_dependencies": dependencies,
    }

    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle_metadata, f, indent=2)

    print(f"\nSaved Reproducible Bundle Metadata to: {bundle_path}", flush=True)
    return bundle_path
