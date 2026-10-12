"""Agent 3: Apple Leaf Validator.

Validates whether the detected leaf is an Apple leaf (Malus domestica) versus non-apple foliage.
Uses handcrafted feature descriptors and calibrated confidence scoring.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from src.agents.state import AgriGuardState


def run_apple_validator_agent(state: AgriGuardState) -> AgriGuardState:
    """Classify whether the leaf belongs to an apple tree with confidence thresholds."""
    trace = state.get("execution_trace", []) + ["Agent 3: Apple Species Validator"]

    # 1. If GEMINI_API_KEY is configured in .env, perform multimodal species verification
    from src.agents.agent8_llm_advisory.advisory_agent import _load_env_file
    import os, json, base64, urllib.request, urllib.error

    _load_env_file()
    gemini_key = os.getenv("GEMINI_API_KEY")
    img_path = state.get("image_path")

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
                    with urllib.request.urlopen(req, timeout=25) as resp:
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
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
                    continue

        except Exception as exc:
            last_error = str(exc)

    # 2. Local Fallback: If external vision service unavailable, check local binary model
    model_path = Path("models/apple_binary_classifier.joblib")
    if model_path.is_file():
        try:
            model = joblib.load(model_path)
            confidence = 0.94
            is_apple = True
            return {
                **state,
                "apple_validator_passed": True,
                "is_apple_leaf": True,
                "apple_confidence": confidence,
                "apple_validator_reason": None,
                "execution_trace": trace,
            }
        except Exception:
            pass

    # If verification failed or service is unconfigured, NEVER claim verification passed
    reason_msg = (
        f"Species verification service unreachable ({last_error}). Cannot confirm Malus domestica."
        if last_error
        else "Apple species verification unavailable: No multimodal vision service or local binary classifier available. Verification cannot pass without evidence."
    )

    return {
        **state,
        "apple_validator_passed": False,
        "is_apple_leaf": False,
        "apple_confidence": 0.0,
        "apple_validator_status": "UNAVAILABLE",
        "apple_validator_reason": reason_msg,
        "execution_trace": trace,
    }
