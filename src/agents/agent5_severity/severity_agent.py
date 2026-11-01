"""Agent 5: Disease Severity Estimation.

Calculates visible lesion coverage percentage:
    visible_percentage = (lesion_area_px / total_leaf_area_px) * 100

Clearly distinguishes visible leaf lesion coverage from whole-orchard yield loss.
Computes pixel-level lesion masks using multi-space OpenCV segmentation.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.agents.state import AgriGuardState


def run_severity_agent(state: AgriGuardState) -> AgriGuardState:
    """Evaluate objective disease severity using OpenCV lesion-to-leaf pixel analysis."""
    trace = state.get("execution_trace", []) + ["Agent 5: Severity Estimation (OpenCV)"]
    img = state.get("image_bgr")
    leaf_mask = state.get("leaf_mask")

    if img is None:
        return {
            **state,
            "severity_percentage": 0.0,
            "severity_tier": "Unavailable (No image)",
            "segmentation_reliable": False,
            "severity_status": "UNAVAILABLE",
            "execution_trace": trace,
        }

    # Standardize image to standardized size
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # If leaf_mask is missing or wrong shape, regenerate it
    if leaf_mask is None or leaf_mask.shape != (256, 256):
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        leaf_mask = cv2.inRange(
            hsv, np.array([12, 25, 25], dtype=np.uint8), np.array([98, 255, 255], dtype=np.uint8)
        )

    total_leaf_pixels = int(cv2.countNonZero(leaf_mask))
    if total_leaf_pixels < 200:
        return {
            **state,
            "severity_percentage": 0.0,
            "severity_tier": "Indeterminate (Leaf area too small)",
            "segmentation_reliable": False,
            "severity_status": "INDETERMINATE",
            "execution_trace": trace,
        }

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 1. Necrotic & dark fungal lesions (Scab, Brown spots, Alternaria, Black rot)
    necrotic_dark = cv2.inRange(
        hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 65], dtype=np.uint8)
    )

    # 2. Orange / reddish rust pustules
    rust_orange = cv2.inRange(
        hsv, np.array([5, 70, 65], dtype=np.uint8), np.array([22, 255, 220], dtype=np.uint8)
    )

    # 3. Chlorotic lesions & yellow halo tissue
    chlorotic_halo = cv2.inRange(
        hsv, np.array([23, 120, 80], dtype=np.uint8), np.array([32, 255, 255], dtype=np.uint8)
    )

    # 4. Powdery mildew white fungal mycelium
    powdery_white = cv2.inRange(
        hsv, np.array([0, 0, 190], dtype=np.uint8), np.array([180, 25, 255], dtype=np.uint8)
    )

    combined_lesions = cv2.bitwise_or(necrotic_dark, rust_orange)
    combined_lesions = cv2.bitwise_or(combined_lesions, chlorotic_halo)
    combined_lesions = cv2.bitwise_or(combined_lesions, powdery_white)

    # Restrict strictly to interior of segmented leaf
    lesion_in_leaf = cv2.bitwise_and(combined_lesions, leaf_mask)

    # Morphological opening to filter micro-shadow artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_lesions = cv2.morphologyEx(lesion_in_leaf, cv2.MORPH_OPEN, kernel)

    lesion_pixels = int(cv2.countNonZero(cleaned_lesions))
    lesion_pct = round(float((lesion_pixels / max(total_leaf_pixels, 1)) * 100.0), 2)

    # Severity Tier Classification based on plant pathology damage index
    if lesion_pct < 1.0:
        tier = "Healthy / Trace (< 1%)"
    elif lesion_pct < 10.0:
        tier = f"Mild / Stage 1 ({lesion_pct:.1f}% foliar coverage)"
    elif lesion_pct < 25.0:
        tier = f"Moderate / Stage 2 ({lesion_pct:.1f}% foliar coverage)"
    else:
        tier = f"Severe / Stage 3 ({lesion_pct:.1f}% foliar coverage)"

    note = (
        f"Visible foliar lesion coverage is {lesion_pct:.1f}% ({tier}). "
        f"Localized leaf lesions represent foliar infection density, not whole-tree yield loss."
    )

    return {
        **state,
        "severity_percentage": lesion_pct,
        "severity_tier": tier,
        "lesion_mask": cleaned_lesions,
        "segmentation_reliable": True,
        "severity_status": "QUANTIFIED",
        "severity_note": note,
        "execution_trace": trace,
    }
