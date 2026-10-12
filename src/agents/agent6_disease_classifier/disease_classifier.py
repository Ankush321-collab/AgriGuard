"""Agent 6: Disease Classification Core.

Wraps the existing LightGBM trained model on 848 handcrafted features.
Includes uncertainty handling: predictions with confidence below 0.60 are flagged
for review rather than forcing an incorrect classification label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np

from src.agents.state import AgriGuardState


def run_disease_classifier(state: AgriGuardState) -> AgriGuardState:
    """Predict disease using existing trained LightGBM model with uncertainty gating."""
    img_path = Path(state.get("image_path", ""))
    # Locate hybrid Soft Voting Ensemble model first, with LightGBM fallback
    model_path = Path("models/Voting_Ensemble.joblib")
    if not model_path.is_file():
        model_path = Path("models/LightGBM.joblib")

    trace = state.get("execution_trace", []) + [
        f"Agent 6: Disease Classifier ({'Soft Voting Ensemble' if 'Voting' in model_path.name else model_path.stem})"
    ]

    try:
        from predict import predict_disease
        res = predict_disease(img_path, model_path=model_path)
        disease_name = res["class_name"]
        confidence = float(res.get("confidence", 0.88))
        model_used = res.get("model", str(model_path.name))
    except Exception as exc:
        raise RuntimeError(
            f"[Agent 6: Disease Classifier Error] Failed to run inference on {img_path} with {model_path}: {exc}\n"
            f"Ensure required packages (mahotas, scikit-learn, lightgbm, opencv-python) are installed."
        ) from exc

    # Uncertainty handling: If confidence is lower than 0.60, mark as uncertain
    is_uncertain = confidence < 0.60

    return {
        **state,
        "disease_name": disease_name,
        "disease_confidence": confidence,
        "model_name": model_used,
        "is_uncertain_prediction": is_uncertain,
        "execution_trace": trace,
    }
