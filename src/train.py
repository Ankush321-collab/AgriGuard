"""Model training and hyperparameter tuning module for AgriGuard-ML.

Implements:
1. Leak-free StandardScaler fitting on the train split only.
2. RandomizedSearchCV hyperparameter tuning for KNN, SVM, LightGBM, and Random Forest.
3. Model evaluation on validation and test splits.
4. Model serialization to models/ with full metadata logging.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.2


def load_dataset(embeddings_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load features, labels, class names, and file paths from HDF5."""
    features_path = embeddings_dir / "features" / "features.h5"
    labels_path = embeddings_dir / "labels" / "labels.h5"

    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Embeddings not found in {embeddings_dir}. Run feature extraction first.")

    with h5py.File(features_path, "r") as file:
        features = file["features"][:]
        paths = file["paths"][:]
    with h5py.File(labels_path, "r") as file:
        labels = file["labels"][:]
        class_names = [name.decode() if isinstance(name, bytes) else str(name) for name in file["class_names"][:]]

    return features, labels, class_names, paths


def prepare_splits(
    features: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    class_names: list[str],
    results_dir: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder
]:
    """Split data into train (70%), val (10%), test (20%) and fit StandardScaler on train ONLY."""
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
            path_str = paths[idx].decode() if isinstance(paths[idx], bytes) else str(paths[idx])
            split_rows.append({
                "feature_row": int(idx),
                "path": path_str,
                "label": int(labels[idx]),
                "class_name": class_names[int(labels[idx])],
                "split": split_name,
            })
    manifest_df = pd.DataFrame(split_rows).sort_values("feature_row")
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(results_dir / "split_manifest.csv", index=False)

    # Fit scaler ONLY on training data to avoid data leakage
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


def tune_and_train_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_names: list[str],
    results_dir: Path,
    device: str = "cpu",
    n_iter: int = 4,
    retune: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune hyperparameters using RandomizedSearchCV (or load cached best params) and fit models."""
    tuning_log_file = results_dir / "hyperparameter_tuning_results.json"
    cached_tuning = {}
    if not retune and tuning_log_file.is_file():
        try:
            with open(tuning_log_file, "r", encoding="utf-8") as f:
                cached_tuning = json.load(f)
            print(f"Loaded existing RandomizedSearchCV tuning results from {tuning_log_file}", flush=True)
        except Exception:
            cached_tuning = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    num_classes = len(class_names)
    best_models: dict[str, Any] = {}
    tuning_log: dict[str, Any] = dict(cached_tuning)

    # Subsample for fast SVM CV exploration
    if len(x_train) > 2500:
        svm_tune_idx, _ = train_test_split(
            np.arange(len(x_train)),
            train_size=2500,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
        x_train_svm_tune = x_train[svm_tune_idx]
        y_train_svm_tune = y_train[svm_tune_idx]
    else:
        x_train_svm_tune = x_train
        y_train_svm_tune = y_train

    search_configs = {
        "KNN": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["distance", "uniform"],
                "metric": ["euclidean", "manhattan"],
            },
            "train_x": x_train,
            "train_y": y_train,
        },
        "SVM": {
            "estimator": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {
                "C": [0.5, 1.0, 5.0, 10.0],
                "gamma": ["scale", "auto", 0.01, 0.005],
            },
            "train_x": x_train_svm_tune,
            "train_y": y_train_svm_tune,
        },
        "LightGBM": {
            "estimator": lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                class_weight="balanced",
                verbosity=-1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [20, 31, 50],
                "max_depth": [-1, 6, 10],
                "subsample": [0.8, 1.0],
            },
            "train_x": x_train,
            "train_y": y_train,
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [None, 15, 25],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            },
            "train_x": x_train,
            "train_y": y_train,
        },
    }

    print("\n" + "=" * 60, flush=True)
    print("Beginning Model Training with Tuned Hyperparameters", flush=True)
    print("=" * 60, flush=True)

    for name, config in search_configs.items():
        if name in cached_tuning and "best_params" in cached_tuning[name] and not retune:
            best_params = cached_tuning[name]["best_params"]
            best_cv_score = cached_tuning[name].get("best_cv_f1_macro", 0.0)
            print(f"\n[Using Tuned {name}] Best params: {best_params} (CV F1: {best_cv_score:.4f})", flush=True)
            # Instantiate with best params
            if name == "KNN":
                model = KNeighborsClassifier(**best_params, n_jobs=-1)
            elif name == "SVM":
                model = SVC(**best_params, kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
            elif name == "LightGBM":
                model = lgb.LGBMClassifier(
                    **best_params,
                    objective="multiclass",
                    num_class=num_classes,
                    class_weight="balanced",
                    verbosity=-1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            elif name == "Random_Forest":
                model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        else:
            print(f"\n[Tuning {name}] running RandomizedSearchCV (n_iter={n_iter}, cv=3)...", flush=True)
            search = RandomizedSearchCV(
                estimator=config["estimator"],
                param_distributions=config["params"],
                n_iter=n_iter,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(config["train_x"], config["train_y"])
            tune_time = time.perf_counter() - started
            best_params = search.best_params_
            best_cv_score = float(search.best_score_)
            print(f"  Best params for {name}: {best_params}", flush=True)
            print(f"  CV Macro-F1: {best_cv_score:.4f} (tuned in {tune_time:.1f}s)", flush=True)
            model = search.best_estimator_
            tuning_log[name] = {
                "best_params": best_params,
                "best_cv_f1_macro": best_cv_score,
                "tune_time_seconds": tune_time,
            }

        # Fit model on FULL training set
        print(f"  Fitting {name} on full training set ({len(x_train)} samples)...", flush=True)
        fit_started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        fit_time = time.perf_counter() - fit_started
        print(f"  Fit complete in {fit_time:.1f}s", flush=True)

        best_models[name] = model
        if name in tuning_log:
            tuning_log[name]["fit_time_seconds"] = fit_time

    # Add Logistic Regression baseline
    print("\n[Fitting Baseline] Logistic Regression...", flush=True)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(x_train, y_train)
    best_models["Logistic_Regression"] = lr
    print("  Logistic Regression fit complete.", flush=True)

    with open(tuning_log_file, "w", encoding="utf-8") as f:
        json.dump(tuning_log, f, indent=2)

    return best_models, tuning_log
