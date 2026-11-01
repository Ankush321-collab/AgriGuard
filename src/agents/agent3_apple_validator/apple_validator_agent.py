"""Agent 3: Apple Leaf Validator.

Validates whether the detected leaf is an Apple leaf (Malus domestica) versus non-apple foliage.
Uses trained calibrated ML classifier on 848 handcrafted features with multimodal vision support.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import joblib
import numpy as np

from src.agents.state import AgriGuardState


def run_apple_validator_agent(state: AgriGuardState) -> AgriGuardState:
    """Classify whether the leaf belongs to an apple tree with confidence thresholds."""
    trace = state.get("execution_trace", []) + ["Agent 3: Apple Species Validator"]
    img_path = state.get("image_path")

    # 1. Primary: Evaluate trained calibrated ML model on 848 handcrafted features
    model_path = Path("models/apple_binary_classifier.joblib")
    if model_path.is_file() and img_path and Path(img_path).is_file():
        try:
            from step1_feature_extraction import image_features
            feat = image_features(img_path)
            if feat is not None:
                model = joblib.load(model_path)
                proba = model.predict_proba(feat.reshape(1, -1))[0]
                apple_conf = float(proba[1])
                is_apple = bool(apple_conf >= 0.50)

                if not is_apple:
                    return {
                        **state,
                        "apple_validator_passed": False,
                        "is_apple_leaf": False,
                        "apple_confidence": apple_conf,
                        "apple_validator_reason": (
                            f"Non-apple foliage detected by 848-feature species model "
                            f"(Malus domestica probability: {apple_conf*100:.1f}%)."
                        ),
                        "execution_trace": trace,
                    }

                return {
                    **state,
                    "apple_validator_passed": True,
                    "is_apple_leaf": True,
                    "apple_confidence": apple_conf,
                    "apple_validator_reason": None,
                    "execution_trace": trace,
                }
        except Exception as exc:
            pass

    # 2. Secondary Multimodal Vision: If local model fails, query Gemini Vision
    from src.agents.agent8_llm_advisory.advisory_agent import _load_env_file
    _load_env_file()
    gemini_key = os.getenv("GEMINI_API_KEY")

    CANDIDATE_MODELS = [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
        "gemini-pro-latest",
    ]

    last_error = None
    if gemini_key and gemini_key != "your_gemini_api_key_here" and img_path and Path(img_path).is_file():
        try:
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            body = {
                "contents": [{
                    "parts": [
                        {
                            "text": (
                                "Inspect this leaf carefully. Is it an apple leaf (Malus domestica) or from another plant/crop/fruit species? "
                                "Respond strictly in JSON: "
                                "{\"is_apple_leaf\": bool, \"detected_species\": str, \"confidence\": float, \"reason\": str}"
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64_data
                            }
                        }
                    ]
                }],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            body_bytes = json.dumps(body).encode("utf-8")

            for model_name in CANDIDATE_MODELS:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                req = urllib.request.Request(
                    url, data=body_bytes, headers={"Content-Type": "application/json"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        parsed = json.loads(resp.read().decode("utf-8"))
                        result_text = parsed["candidates"][0]["content"]["parts"][0]["text"]
                        spec_data = json.loads(result_text)

                        is_apple = bool(spec_data.get("is_apple_leaf", False))
                        confidence = float(spec_data.get("confidence", 0.90))
                        species = spec_data.get("detected_species", "Unknown species")
                        reason = spec_data.get("reason", "")

                        if not is_apple or confidence < 0.60:
                            return {
                                **state,
                                "apple_validator_passed": False,
                                "is_apple_leaf": False,
                                "apple_confidence": confidence,
                                "apple_validator_reason": f"Non-apple leaf detected: {species} (Confidence: {confidence*100:.1f}%). {reason}",
                                "execution_trace": trace,
                            }

                        return {
                            **state,
                            "apple_validator_passed": True,
                            "is_apple_leaf": True,
                            "apple_confidence": confidence,
                            "apple_validator_reason": None,
                            "execution_trace": trace,
                        }
                except Exception as exc:
                    last_error = str(exc)
                    continue
        except Exception as exc:
            last_error = str(exc)

    return {
        **state,
        "apple_validator_passed": False,
        "is_apple_leaf": False,
        "apple_confidence": 0.0,
        "apple_validator_status": "UNAVAILABLE",
        "apple_validator_reason": (
            f"Apple species verification failed: {last_error or 'No model available'}."
        ),
        "execution_trace": trace,
    }
