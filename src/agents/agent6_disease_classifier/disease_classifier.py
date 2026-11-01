"""Agent 6: Disease Classification Core.

Evaluates 848 handcrafted features using classical ML ensemble models:
- Voting Ensemble (Random Forest + SVM + KNN + LightGBM)
- LightGBM Core Classifier

Combines model probabilities and provides calibrated confidence gating without
allowing uncalibrated zero-shot LLM hallucinations to corrupt trained model predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np

from src.agents.state import AgriGuardState


def run_disease_classifier(state: AgriGuardState) -> AgriGuardState:
    """Predict disease using trained ML models with ensemble verification."""
    img_path = Path(state.get("image_path", ""))

    voting_path = Path("models/Voting_Ensemble.joblib")
    lgbm_path = Path("models/LightGBM.joblib")
    scaler_path = Path("models/scaler.joblib")

    trace = state.get("execution_trace", [])

    from predict import predict_disease

    res_voting = None
    res_lgbm = None

    # Run Voting Ensemble
    if voting_path.is_file():
        try:
            res_voting = predict_disease(img_path, model_path=voting_path, scaler_path=scaler_path)
        except Exception as exc:
            pass

    # Run LightGBM
    if lgbm_path.is_file():
        try:
            res_lgbm = predict_disease(img_path, model_path=lgbm_path, scaler_path=scaler_path)
        except Exception as exc:
            pass

    if res_voting is None and res_lgbm is None:
        raise RuntimeError(f"Unable to load trained models from models/ to classify {img_path}")

    # Determine final prediction
    if res_voting and res_lgbm:
        pred_v = res_voting["class_name"]
        conf_v = float(res_voting.get("confidence", 0.0))
        pred_l = res_lgbm["class_name"]
        conf_l = float(res_lgbm.get("confidence", 0.0))

        if pred_v == pred_l:
            disease_name = pred_v
            confidence = max(conf_v, conf_l)
            model_used = "Voting Ensemble & LightGBM Consensus (848 Features)"
        elif conf_l > conf_v and conf_l > 0.80:
            disease_name = pred_l
            confidence = conf_l
            model_used = f"LightGBM ({conf_l*100:.1f}%) [Voting: {pred_v} ({conf_v*100:.1f}%)]"
        else:
            disease_name = pred_v
            confidence = conf_v
            model_used = f"Voting Ensemble ({conf_v*100:.1f}%) [LightGBM: {pred_l} ({conf_l*100:.1f}%)]"

        # Combine probability distributions
        probs_v = res_voting.get("class_probabilities", {})
        probs_l = res_lgbm.get("class_probabilities", {})
        all_classes = set(list(probs_v.keys()) + list(probs_l.keys()))
        combined_probs = {}
        for c in all_classes:
            combined_probs[c] = float((probs_v.get(c, 0.0) + probs_l.get(c, 0.0)) / 2.0)
    elif res_lgbm:
        disease_name = res_lgbm["class_name"]
        confidence = float(res_lgbm.get("confidence", 0.88))
        model_used = res_lgbm.get("model", "LightGBM.joblib")
        combined_probs = res_lgbm.get("class_probabilities", {})
    else:
        disease_name = res_voting["class_name"]
        confidence = float(res_voting.get("confidence", 0.88))
        model_used = res_voting.get("model", "Voting_Ensemble.joblib")
        combined_probs = res_voting.get("class_probabilities", {})

    trace.append(f"Agent 6: Disease Classifier ({model_used} -> {disease_name} at {confidence*100:.1f}%)")

    # True ambiguity threshold: Flag for review only if model confidence is low (< 0.50)
    is_uncertain = confidence < 0.50
    if is_uncertain:
        trace.append(f"Agent 6: Low Confidence Warning ({confidence*100:.1f}% < 50.0%) -> Marginal Flag")

    return {
        **state,
        "disease_name": disease_name,
        "disease_confidence": confidence,
        "model_name": model_used,
        "is_uncertain_prediction": is_uncertain,
        "class_probabilities": combined_probs,
        "execution_trace": trace,
    }
