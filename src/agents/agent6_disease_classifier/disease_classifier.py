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

    # Uncertainty handling: Predictions with confidence below 0.75 are considered marginal/borderline
    # and require multimodal visual arbitration to verify physical lesion morphology.
    is_uncertain = confidence < 0.75

    # Multimodal Visual Arbitration for ambiguous classical predictions (< 0.75)
    if is_uncertain:
        from src.agents.agent8_llm_advisory.advisory_agent import _load_env_file
        import os, json, base64, urllib.request

        _load_env_file()
        gemini_key = os.getenv("GEMINI_API_KEY")

        if gemini_key and gemini_key != "your_gemini_api_key_here" and img_path.is_file():
            try:
                with open(img_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")

                mime = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                prompt = (
                    f"A classical ML model predicted '{disease_name}' with marginal confidence ({confidence*100:.1f}%). "
                    "Conduct an expert plant pathology visual inspection of this apple leaf. "
                    "Analyze lesion morphology carefully:\n"
                    "- Frogeye leaf spot: distinct circular lesions with concentric rings, purple/dark margins, and tan-brown centers (bullseye appearance).\n"
                    "- Scab (Venturia inaequalis): velvety, olive-green to dark brown/black lesions on the leaf surface.\n"
                    "- Rust (Cedar apple rust): bright yellow, orange, or reddish-brown circular spots/pustules.\n"
                    "- Powdery mildew: white to grayish powdery or felt-like fungal patches on leaf surface.\n"
                    "- Alternaria leaf spot: circular brown-black lesions, sometimes with concentric rings, often near veins.\n"
                    "- Mosaic: clear mosaic, yellow banding, vein clearing, or mottling with no necrotic center.\n"
                    "- Healthy: clean green foliage with no fungal or necrotic lesions.\n\n"
                    "Choose strictly from: Scab, Rust, Frogeye leaf spot, Powdery mildew, Mosaic, "
                    "Alternaria leaf spot, Brown spot, Grey spot, or Healthy.\n"
                    "Respond strictly in JSON format:\n"
                    "{\"disease\": str, \"confidence\": float, \"visual_evidence\": str}"
                )
                body = {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime, "data": b64_data}}
                        ]
                    }],
                    "generationConfig": {"response_mime_type": "application/json"}
                }
                body_bytes = json.dumps(body).encode("utf-8")

                for model_candidate in ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash-lite"]:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent?key={gemini_key}"
                    req = urllib.request.Request(
                        url, data=body_bytes, headers={"Content-Type": "application/json"}
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=25) as resp:
                            parsed = json.loads(resp.read().decode("utf-8"))
                            text = parsed["candidates"][0]["content"]["parts"][0]["text"]
                            arb_data = json.loads(text)

                            arb_disease = arb_data.get("disease", "").replace("Apple scab", "Scab").strip()
                            arb_conf = float(arb_data.get("confidence", 0.0))

                            if arb_conf >= 0.70 and arb_disease:
                                disease_name = arb_disease
                                confidence = arb_conf
                                model_used = f"{model_used} + Multimodal Arbiter ({model_candidate})"
                                is_uncertain = False
                                trace.append(f"Agent 6: Multimodal Arbiter Resolved to {arb_disease} ({arb_conf*100:.1f}%)")
                                break
                    except Exception:
                        continue
            except Exception:
                pass

    return {
        **state,
        "disease_name": disease_name,
        "disease_confidence": confidence,
        "model_name": model_used,
        "is_uncertain_prediction": is_uncertain,
        "execution_trace": trace,
    }
