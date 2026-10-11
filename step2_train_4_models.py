"""Train, tune, and evaluate classical ML models and ensembles on extracted features.

Features:
1. Leak-free StandardScaler fitted strictly on the train split.
2. Hyperparameter tuning using RandomizedSearchCV across KNN, SVM, LightGBM, and Random Forest.
3. Soft Voting Ensemble and Stacking Ensemble (Logistic Regression meta-learner).
4. Validation and test evaluation, confusion matrices, and best model selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ensemble import build_and_train_ensembles, evaluate_all_models
from src.evaluate import plot_confusion_matrices, plot_metrics_comparison
from src.train import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    load_dataset,
    prepare_splits,
    tune_and_train_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgriGuard-ML Model Training & Ensemble Pipeline")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--tune-iter", type=int, default=5, help="Number of RandomizedSearchCV iterations per model")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print(f"Loading feature embeddings from {args.embeddings_dir}...")
    features, labels, class_names, paths = load_dataset(args.embeddings_dir)
    print(f"Loaded {features.shape[0]} samples with feature dimension {features.shape[1]}.")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    # 2. Prepare train/val/test splits and fit scaler on TRAIN ONLY
    print("\nPreparing stratified splits (70% train, 10% val, 20% test)...")
    (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        train_indices, val_indices, test_indices,
        scaler, label_encoder,
    ) = prepare_splits(features, labels, paths, class_names, args.results_dir)

    print(f"Split counts: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")

    # Save standalone scaler and label encoder as requested in README
    joblib.dump(scaler, args.models_dir / "scaler.joblib")
    joblib.dump(label_encoder, args.models_dir / "label_encoder.joblib")
    print("Saved StandardScaler and LabelEncoder to models/ (train-only fit).")

    # 3. Hyperparameter tuning and base model training
    base_models, tuning_log = tune_and_train_models(
        x_train=x_train,
        y_train=y_train,
        class_names=class_names,
        results_dir=args.results_dir,
        device=args.device,
        n_iter=args.tune_iter,
    )

    with open(args.results_dir / "hyperparameter_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(tuning_log, f, indent=2)
    print(f"Logged tuning results to {args.results_dir / 'hyperparameter_tuning_results.json'}")

    # 4. Ensembles (Soft Voting & Stacking)
    ensemble_models = build_and_train_ensembles(
        base_models=base_models,
        x_train=x_train,
        y_train=y_train,
    )

    # Combine all models for evaluation
    all_models = {**base_models, **ensemble_models}

    # 5. Evaluate all models on validation and test sets
    comparison_df, best_selection, confusion_matrices = evaluate_all_models(
        all_models=all_models,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
    )

    # 6. Plot visualizations
    plot_confusion_matrices(
        confusion_matrices=confusion_matrices,
        class_names=class_names,
        output_path=args.results_dir / "confusion_matrices_all_models.png",
    )
    plot_metrics_comparison(
        comparison_df=comparison_df,
        output_path=args.results_dir / "model_metrics_comparison.png",
    )

    # 7. Save comprehensive training metadata
    metadata = {
        "feature_count": int(features.shape[1]),
        "feature_pipeline_version": "appleleaf9-full-multimodal-v2",
        "segmentation": "exg-hsv-morphology-contour",
        "feature_families": ["color_524", "texture_haralick_lbp_23", "shape_hu_contour_13", "gradient_hog_288"],
        "class_names": class_names,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "train_count": len(train_indices),
        "validation_count": len(val_indices),
        "test_count": len(test_indices),
        "best_model_by_val_macro_f1": best_selection["best_model_name"],
        "best_val_macro_f1": best_selection["validation_macro_f1"],
        "best_test_macro_f1": best_selection["test_macro_f1"],
        "best_test_accuracy": best_selection["test_accuracy"],
        "models_evaluated": list(all_models.keys()),
        "hyperparameter_tuning": "RandomizedSearchCV with 3-fold StratifiedKFold",
        "ensembles_implemented": ["Voting_Ensemble", "Stacking_Ensemble"],
    }
    with open(args.results_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING & ENSEMBLE SUMMARY")
    print("=" * 60)
    print(comparison_df[["Model", "Validation-Macro-F1", "Macro-F1", "Accuracy", "InferenceTimeSeconds"]].to_string(index=False))
    print(f"\nWinner selected: {best_selection['best_model_name']} (Val Macro-F1: {best_selection['validation_macro_f1']:.4f})")
    print(f"Artifacts saved in {args.models_dir} and {args.results_dir}")


if __name__ == "__main__":
    main()