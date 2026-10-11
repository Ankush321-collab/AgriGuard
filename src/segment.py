"""Leaf segmentation module for AgriGuard-ML.

Implements color-based HSV thresholding, Excess Green (ExG) index Otsu segmentation,
morphological cleanup, and largest-contour extraction to separate the leaf
from background and compute explainable geometric shape descriptors.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def segment_leaf(
    image: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    min_area_fraction: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Segment the primary leaf from the background and compute shape metrics.

    Args:
        image: BGR image as a NumPy array.
        target_size: Output dimensions (width, height) to standardize to.
        min_area_fraction: Minimum contour area fraction relative to image area
            required to consider a detected contour a valid leaf.

    Returns:
        tuple containing:
            - segmented_image: BGR image with background masked out to black.
            - mask: 2D uint8 binary mask (255 = leaf, 0 = background).
            - shape_metrics: Dictionary of explainable shape features.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image provided to segment_leaf.")

    # 1. Resize to standardized dimensions
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    height, width = target_size[1], target_size[0]
    total_area = float(height * width)

    # 2. Excess Green (ExG) Index: 2*G - R - B
    # Highly effective agricultural index for separating plant foliage from backgrounds.
    b, g, r = cv2.split(resized.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh_exg = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. HSV foliage color thresholding (captures green, yellowish, and brownish diseased foliage)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower_foliage = np.array([12, 25, 25], dtype=np.uint8)
    upper_foliage = np.array([98, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_foliage, upper_foliage)

    # 4. Combine ExG Otsu and HSV foliage masks
    combined_mask = cv2.bitwise_or(thresh_exg, mask_hsv)

    # 5. Morphological operations
    # Close: fills internal holes and necrotic disease lesions inside the leaf boundary
    # Open: removes small extraneous background artifacts
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 6. Extract external contours and select the largest contour
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_mask = np.zeros((height, width), dtype=np.uint8)
    shape_metrics: dict[str, float] = {}

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(largest_contour))

        if contour_area >= (total_area * min_area_fraction):
            cv2.drawContours(leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            perimeter = float(cv2.arcLength(largest_contour, True))
            circularity = (4.0 * np.pi * contour_area) / (perimeter**2 + 1e-7)

            bx, by, bw, bh = cv2.boundingRect(largest_contour)
            aspect_ratio = float(bw) / float(max(bh, 1))
            extent = float(contour_area) / float(max(bw * bh, 1))

            hull = cv2.convexHull(largest_contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(contour_area) / float(max(hull_area, 1.0))

            shape_metrics = {
                "area": contour_area,
                "perimeter": perimeter,
                "area_ratio": contour_area / total_area,
                "perimeter_ratio": perimeter / (2.0 * (height + width)),
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "solidity": solidity,
            }
        else:
            # Contour is too small - fallback to full mask
            leaf_mask.fill(255)
            shape_metrics = {
                "area": total_area,
                "perimeter": 2.0 * (height + width),
                "area_ratio": 1.0,
                "perimeter_ratio": 1.0,
                "circularity": 0.5,
                "aspect_ratio": 1.0,
                "extent": 1.0,
                "solidity": 1.0,
            }
    else:
        # Fallback if no contours detected
        leaf_mask.fill(255)
        shape_metrics = {
            "area": total_area,
            "perimeter": 2.0 * (height + width),
            "area_ratio": 1.0,
            "perimeter_ratio": 1.0,
            "circularity": 0.5,
            "aspect_ratio": 1.0,
            "extent": 1.0,
            "solidity": 1.0,
        }

    # 7. Apply mask to produce segmented leaf image
    segmented_image = cv2.bitwise_and(resized, resized, mask=leaf_mask)

    return segmented_image, leaf_mask, shape_metrics
