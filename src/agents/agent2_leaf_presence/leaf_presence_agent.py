"""Agent 2: Leaf Presence Agent.

Checks whether the frame contains a valid plant foliage region using Excess Green index (ExG),
HSV foliage color thresholding, and morphological contour/texture validation.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from src.agents.state import AgriGuardState


def run_leaf_presence_agent(state: AgriGuardState) -> AgriGuardState:
    """Verify that a plant leaf region is present in the foreground."""
    trace = state.get("execution_trace", []) + ["Agent 2: Leaf Presence Check"]
    img = state.get("image_bgr")

    if img is None:
        img_path = state.get("image_path")
        if img_path and Path(img_path).is_file():
            img = cv2.imread(str(img_path))

    if img is None:
        return {
            **state,
            "leaf_presence_passed": False,
            "leaf_presence_reason": "No image available for leaf presence analysis.",
            "execution_trace": trace,
        }

    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # 1. Excess Green (ExG) Index: 2G - R - B
    b, g, r = cv2.split(resized.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_ratio = float(np.sum(exg > 10.0)) / (256.0 * 256.0)

    # 2. HSV foliage mask (healthy greens, yellow chlorosis, brown lesions)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    foliage_mask = cv2.inRange(
        hsv, np.array([12, 25, 25], dtype=np.uint8), np.array([98, 255, 255], dtype=np.uint8)
    )
    foliage_ratio = float(cv2.countNonZero(foliage_mask)) / (256.0 * 256.0)

    # 3. Micro-texture variance: Real biological leaves have venation/edges, flat walls/desks do not
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Vegetative presence check:
    # A leaf must contain either active green chlorophyll (exg_ratio >= 0.08)
    # OR a combination of foliage coloration (foliage_ratio >= 0.12) with natural biological texture (texture_var >= 10.0).
    has_chlorophyll = bool(exg_ratio >= 0.08)
    has_textured_foliage = bool(foliage_ratio >= 0.12 and texture_var >= 10.0)

    if not has_chlorophyll and not has_textured_foliage:
        return {
            **state,
            "leaf_presence_passed": False,
            "leaf_area_fraction": max(foliage_ratio, exg_ratio),
            "leaf_mask": foliage_mask,
            "leaf_presence_reason": "No plant leaf detected. Frame lacks vegetative chlorophyll or natural leaf texture.",
            "execution_trace": trace,
        }

    return {
        **state,
        "leaf_presence_passed": True,
        "leaf_area_fraction": max(foliage_ratio, exg_ratio),
        "leaf_mask": foliage_mask,
        "leaf_presence_reason": None,
        "execution_trace": trace,
    }
