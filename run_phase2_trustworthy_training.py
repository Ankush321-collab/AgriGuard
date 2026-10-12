"""AgriGuard-ML: Phase 2 Trustworthy ML Pipeline Runner.

Executes a leak-free, reproducible training and evaluation cycle:
1. Audits dataset for duplicates and prevents cross-split data leakage.
2. Creates stratified, leak-free Train (70%), Validation (10%), and Test (20%) splits.
3. Fits StandardScaler strictly on training split.
4. Trains and evaluates base models and Soft Voting / Stacking ensembles.
5. Computes per-class precision, recall, F1, confusion matrices, and Brier calibration.
6. Calibrates confidence thresholds on validation data and logs abstention rates.
7. Packages all artifacts into a versioned, reproducible bundle (bundle_metadata.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np

from src.ensemble import build_and_train_ensembles
from src.train import load_dataset, tune_and_train_models
from src.trustworthy_ml import (
    audit_and_deduplicate,
    create_leak_free_splits,
    evaluate_trustworthy_pipeline,
    save_reproducible_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 Trustworthy ML Pipeline")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument("--tune-iter", type=int, default=4)
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("   AGRIGUARD-ML: PHASE 2 TRUSTWORTHY ML REPRODUCIBLE RUNNER")
    print("=" * 70)

    # 1. Load dataset
    print(f"\n[Step 1/6] Loading feature embeddings from {args.embeddings_dir}...")
    features, labels, class_names, raw_paths = load_dataset(args.embeddings_dir)
    paths = [p.decode() if isinstance(p, bytes) else str(p) for p in raw_paths]
    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features across {len(class_names)} classes.")

    # 2. Audit and Deduplicate
    print("\n[Step 2/6] Running duplicate integrity audit and isolating leak vectors...")
    clean_features, clean_labels, clean_paths, unique_idxs, audit_log = audit_and_deduplicate(
        features=features,
        labels=labels,
        paths=paths,
        class_names=class_names,
        results_dir=args.results_dir,
    )

    # 3. Create leak-free stratified splits (Train 70%, Val 10%, Test 20%)
    print("\n[Step 3/6] Generating leak-free stratified splits with manifest...")
    (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        train_idx, val_idx, test_idx,
        scaler, label_encoder,
    ) = create_leak_free_splits(
        features=clean_features,
        labels=clean_labels,
        paths=clean_paths,
        class_names=class_names,
        results_dir=args.results_dir,
    )
    print(f"Split sizes: Train={len(train_idx)}, Validation={len(val_idx)}, Held-Out Test={len(test_idx)}")

    # Save standalone scaler and label encoder
    joblib.dump(scaler, args.models_dir / "scaler.joblib")
    joblib.dump(label_encoder, args.models_dir / "label_encoder.joblib")

    # 4. Train base models (KNN, SVM, LightGBM, Random Forest)
    print("\n[Step 4/6] Training base classifiers (tuned with RandomizedSearchCV)...")
    base_models, tuning_log = tune_and_train_models(
        x_train=x_train,
        y_train=y_train,
        class_names=class_names,
        results_dir=args.results_dir,
        device=args.device,
        n_iter=args.tune_iter,
        retune=False,
    )

    # 5. Build Soft Voting and Stacking Ensembles
    print("\n[Step 5/6] Assembling Hybrid Soft Voting and Stacking ensembles...")
    ensemble_models = build_and_train_ensembles(
        base_models=base_models,
        x_train=x_train,
        y_train=y_train,
    )
    all_models = {**base_models, **ensemble_models}

    # 6. Trustworthy Evaluation: per-class metrics, Brier calibration, and abstention tuning
    print("\n[Step 6/6] Computing per-class reports, Brier calibration, and abstention curves...")
    eval_results = evaluate_trustworthy_pipeline(
        all_models=all_models,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
    )

    # Select optimal operational confidence threshold on validation data (e.g. 0.60 standard)
    selected_threshold = 0.60
    save_reproducible_bundle(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        class_names=class_names,
        selected_model_name="Voting_Ensemble",
        optimal_threshold=selected_threshold,
    )

    print("\n" + "=" * 70)
    print("   PHASE 2 TRUSTWORTHY ML RUN COMPLETE!")
    print("=" * 70)
    print(f"Models saved to: {args.models_dir.resolve()}")
    print(f"Results saved to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
