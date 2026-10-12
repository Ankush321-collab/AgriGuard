"""Agent 2: Leaf Presence Agent.

Checks whether the frame contains a valid plant foliage region using Excess Green index (ExG)
and HSV foliage color thresholding.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.agents.state import AgriGuardState


def run_leaf_presence_agent(state: AgriGuardState) -> AgriGuardState:
    """Verify that a plant leaf region is present in the foreground."""
    trace = state.get("execution_trace", []) + ["Agent 2: Leaf Presence Check"]
    img = state.get("image_bgr")

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

    min_area_ratio = 0.12
    if foliage_ratio < min_area_ratio and exg_ratio < min_area_ratio:
        return {
            **state,
            "leaf_presence_passed": False,
            "leaf_area_fraction": foliage_ratio,
            "leaf_mask": foliage_mask,
            "leaf_presence_reason": "No clear plant leaf detected. Place a single leaf centered in frame.",
            "execution_trace": trace,
        }

    return {
        **state,
        "leaf_presence_passed": True,
        "leaf_area_fraction": foliage_ratio,
        "leaf_mask": foliage_mask,
        "leaf_presence_reason": None,
        "execution_trace": trace,
    }
