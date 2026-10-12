"""Agent 5: Disease Severity Estimation.

Calculates visible lesion coverage percentage:
    visible_percentage = (lesion_area_px / total_leaf_area_px) * 100

Clearly distinguishes visible leaf lesion coverage from total crop yield loss.
Flags segmentation as unreliable when masks are unstable.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.agents.state import AgriGuardState


def run_severity_agent(state: AgriGuardState) -> AgriGuardState:
    """Measure visible necrotic lesion surface area and map to IPM severity tiers."""
    trace = state.get("execution_trace", []) + ["Agent 5: Severity Estimation"]
    img = state.get("image_bgr")
    leaf_mask = state.get("leaf_mask")

    if img is None or leaf_mask is None:
        return {
            **state,
            "severity_percentage": None,
            "severity_tier": "Unavailable",
            "segmentation_reliable": False,
            "execution_trace": trace,
        }

    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    if total_leaf_pixels < 500:
        return {
            **state,
            "severity_percentage": None,
            "severity_tier": "Unreliable / Leaf Area Too Small",
            "segmentation_reliable": False,
            "execution_trace": trace,
        }

    # Necrotic/chlorotic lesion detection (brown, yellow, dark spots)
    lesion_mask_raw = cv2.inRange(
        hsv, np.array([8, 45, 25], dtype=np.uint8), np.array([32, 255, 185], dtype=np.uint8)
    )
    lesion_mask = cv2.bitwise_and(lesion_mask_raw, lesion_mask_raw, mask=leaf_mask)
    lesion_pixels = cv2.countNonZero(lesion_mask)

    percentage = round((float(lesion_pixels) / float(total_leaf_pixels)) * 100.0, 2)

    # Integrated Pest Management (IPM) Severity Tiers
    if percentage < 5.0:
        tier = "Stage 1: Mild (< 5% visible coverage)"
    elif percentage <= 25.0:
        tier = "Stage 2: Moderate (5% - 25% visible coverage)"
    else:
        tier = "Stage 3: Severe (> 25% visible coverage)"

    return {
        **state,
        "severity_percentage": percentage,
        "severity_tier": tier,
        "lesion_mask": lesion_mask,
        "segmentation_reliable": True,
        "execution_trace": trace,
    }
