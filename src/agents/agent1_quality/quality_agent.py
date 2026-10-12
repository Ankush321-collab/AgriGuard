"""Agent 1: Image Quality Guard.

Validates image readability, blur score (Laplacian variance), illumination, and resolution.
"""

from __future__ import annotations

from typing import Any, Dict
import cv2
import numpy as np

from src.agents.state import AgriGuardState


def run_quality_agent(state: AgriGuardState) -> AgriGuardState:
    """Audit image clarity, lighting, and resolution using deterministic OpenCV."""
    img_path = state.get("image_path", "")
    trace = state.get("execution_trace", []) + ["Agent 1: Quality Check"]

    img = cv2.imread(img_path)
    if img is None:
        return {
            **state,
            "quality_passed": False,
            "quality_reason": f"File could not be decoded as an image: {img_path}",
            "execution_trace": trace,
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resolution-standardized Laplacian focus estimation:
    # High-resolution smooth surfaces (such as healthy foliage with subtle venation)
    # produce naturally low micro-gradients on raw pixel scales. Standardizing to 512px
    # ensures consistent edge/focus variance across resolutions.
    max_dim = max(h, w)
    if max_dim > 512:
        scale = 512.0 / max_dim
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    laplacian_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    # Threshold checks
    if h < 256 or w < 256:
        return {
            **state,
            "image_bgr": img,
            "quality_passed": False,
            "resolution": [w, h],
            "blur_score": laplacian_var,
            "brightness": mean_brightness,
            "quality_reason": f"Resolution ({w}x{h}) is lower than minimum 256x256.",
            "execution_trace": trace,
        }

    if laplacian_var < 8.0:
        return {
            **state,
            "image_bgr": img,
            "quality_passed": False,
            "resolution": [w, h],
            "blur_score": laplacian_var,
            "brightness": mean_brightness,
            "quality_reason": f"Image is blurry (focus score: {laplacian_var:.1f} < 8.0). Hold the camera steady.",
            "execution_trace": trace,
        }

    if mean_brightness < 35.0:
        return {
            **state,
            "image_bgr": img,
            "quality_passed": False,
            "resolution": [w, h],
            "blur_score": laplacian_var,
            "brightness": mean_brightness,
            "quality_reason": f"Image is underexposed/dark (brightness: {mean_brightness:.1f} < 35.0). Capture in daylight.",
            "execution_trace": trace,
        }

    if mean_brightness > 225.0:
        return {
            **state,
            "image_bgr": img,
            "quality_passed": False,
            "resolution": [w, h],
            "blur_score": laplacian_var,
            "brightness": mean_brightness,
            "quality_reason": f"Image has harsh glare/overexposure (brightness: {mean_brightness:.1f} > 225.0).",
            "execution_trace": trace,
        }

    return {
        **state,
        "image_bgr": img,
        "quality_passed": True,
        "resolution": [w, h],
        "blur_score": laplacian_var,
        "brightness": mean_brightness,
        "quality_reason": None,
        "execution_trace": trace,
    }
