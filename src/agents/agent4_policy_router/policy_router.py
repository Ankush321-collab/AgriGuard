"""Agent 4: Policy Router.

Central deterministic gatekeeper evaluating upstream validation states:
- RETAKE: Image quality issue (blurry, dark, overexposed, low resolution).
- REJECT: Outside supported scope (no plant foliage, non-apple crop leaf).
- REVIEW: Uncertain / borderline species confidence (0.50 <= P < 0.75).
- PASS: Fully validated apple leaf image ready for ML Core diagnosis.
"""

from __future__ import annotations

from src.agents.state import AgriGuardState


def run_policy_router(state: AgriGuardState) -> AgriGuardState:
    """Evaluate validation outputs and set policy decision with actionable feedback."""
    trace = state.get("execution_trace", []) + ["Agent 4: Policy Router Gate"]

    # 1. Quality failure -> RETAKE
    if not state.get("quality_passed", False):
        reason = state.get("quality_reason", "Image quality is insufficient.")
        return {
            **state,
            "policy_decision": "RETAKE",
            "policy_reason": reason,
            "suggested_action": "Hold camera steady 15-20cm away in indirect daylight and retake the photo.",
            "execution_trace": trace,
        }

    # 2. No leaf detected -> REJECT
    if not state.get("leaf_presence_passed", False):
        reason = state.get("leaf_presence_reason", "No plant leaf detected in the frame.")
        return {
            **state,
            "policy_decision": "REJECT",
            "policy_reason": reason,
            "suggested_action": "Upload an image with a single plant leaf centered against a neutral background.",
            "execution_trace": trace,
        }

    # 3. Species verification unavailable -> REVIEW
    if state.get("apple_validator_status") == "UNAVAILABLE":
        return {
            **state,
            "policy_decision": "REVIEW",
            "policy_reason": state.get(
                "apple_validator_reason",
                "Apple species verification unavailable: Verification cannot pass without evidence.",
            ),
            "suggested_action": "Request expert agronomist inspection to verify tree species before taking action.",
            "execution_trace": trace,
        }

    # 4. Confirmed Non-apple leaf -> REJECT
    if not state.get("apple_validator_passed", False):
        reason = state.get("apple_validator_reason", "Non-apple crop leaf detected.")
        return {
            **state,
            "policy_decision": "REJECT",
            "policy_reason": reason,
            "suggested_action": "AgriGuard-ML is calibrated for apple orchards. Upload an apple leaf.",
            "execution_trace": trace,
        }

    # 5. Borderline species confidence -> REVIEW
    apple_conf = state.get("apple_confidence", 1.0)
    if 0.50 <= apple_conf < 0.75:
        return {
            **state,
            "policy_decision": "REVIEW",
            "policy_reason": f"Borderline apple leaf confidence ({apple_conf*100:.1f}%).",
            "suggested_action": "Upload a clearer leaf photo or verify foliage type with an agricultural extension agent.",
            "execution_trace": trace,
        }

    # 5. Fully validated -> PASS
    return {
        **state,
        "policy_decision": "PASS",
        "policy_reason": "Image passed all quality, foliage presence, and apple-leaf species checks.",
        "suggested_action": "Proceed to disease classification and severity estimation.",
        "execution_trace": trace,
    }
