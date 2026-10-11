"""Predict apple leaf disease for a single image using a trained model or ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from step1_feature_extraction import image_features


def load_class_names(labels_path: Path) -> list[str]:
    """Load class names from labels HDF5 or default AppleLeaf9 classes."""
    if labels_path.is_file():
        with h5py.File(labels_path, "r") as file:
            return [name.decode() if isinstance(name, bytes) else str(name) for name in file["class_names"][:]]
    return [
        "Alternaria leaf spot",
        "Brown spot",
        "Frogeye leaf spot",
        "Grey spot",
        "Healthy",
        "Mosaic",
        "Powdery mildew",
        "Rust",
        "Scab",
    ]


def predict_disease(
    image_path: Path,
    model_path: Path = Path("models/LightGBM.joblib"),
    scaler_path: Path = Path("models/scaler.joblib"),
    labels_path: Path = Path("embeddings/labels/labels.h5"),
) -> dict[str, Any]:
    """Run leaf segmentation, feature extraction, scaling, and model prediction."""
    features = image_features(image_path)
    if features is None:
        raise ValueError(f"Unable to extract valid 848-feature vector from {image_path}")

    class_names = load_class_names(labels_path)

    if not model_path.is_file():
        # Fallback to .pkl or other candidate
        pkl_path = model_path.with_suffix(".pkl")
        if pkl_path.is_file():
            model_path = pkl_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    feat_2d = features.reshape(1, -1)

    # Scale features if model is not already an end-to-end Pipeline
    is_pipeline = isinstance(model, Pipeline)
    if not is_pipeline and scaler_path.is_file():
        scaler = joblib.load(scaler_path)
        feat_input = scaler.transform(feat_2d)
    else:
        feat_input = feat_2d

    prediction = int(model.predict(feat_input)[0])
    result: dict[str, Any] = {
        "image": str(image_path),
        "model": str(model_path.name),
        "class_id": prediction,
        "class_name": class_names[prediction],
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feat_input)[0]
        result["confidence"] = float(probabilities[prediction])
        result["class_probabilities"] = {
            class_names[i]: float(prob) for i, prob in enumerate(probabilities)
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Apple Leaf Disease")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/LightGBM.joblib"),
        help="Path to trained model or ensemble",
    )
    parser.add_argument("--scaler", type=Path, default=Path("models/scaler.joblib"))
    parser.add_argument("--labels", type=Path, default=Path("embeddings/labels/labels.h5"))
    args = parser.parse_args()

    # If default model doesn't exist yet, fallback to LightGBM or first available
    if not args.model.is_file():
        for candidate in ["LightGBM.joblib", "Random_Forest.joblib", "SVM.joblib", "KNN.joblib"]:
            cand_path = Path("models") / candidate
            if cand_path.is_file():
                args.model = cand_path
                break

    result = predict_disease(
        image_path=args.image,
        model_path=args.model,
        scaler_path=args.scaler,
        labels_path=args.labels,
    )
    print("\n--- AgriGuard-ML Diagnosis Result ---")
    print(f"Image:       {result['image']}")
    print(f"Model Used:  {result['model']}")
    print(f"Prediction:  {result['class_name']} (Class ID: {result['class_id']})")
    if "confidence" in result:
        print(f"Confidence:  {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()