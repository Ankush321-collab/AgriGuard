"""Feature extraction module for AgriGuard-ML.

Implements all four handcrafted feature families specified in README_AgriGuard_ML.md:
1. Color: 3D HSV histogram (512) + RGB mean & std (6) + HSV mean & std (6) = 524 features
2. Texture: GLCM Haralick moments (13) + Uniform Local Binary Patterns (LBP) histogram (10) = 23 features
3. Shape: Log-transformed Hu Moments (7) + Segmented leaf contour geometry (6) = 13 features
4. Gradient: Histogram of Oriented Gradients (HOG) descriptors = 288 features

Total feature vector dimension: 848 features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import cv2
import mahotas
import numpy as np
from skimage.feature import hog, local_binary_pattern

from src.preprocess import preprocess_image

FEATURE_COUNT = 848
FEATURE_PIPELINE_VERSION = "appleleaf9-full-multimodal-v2"


def extract_color_features(hsv: np.ndarray, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract 3D HSV color histogram and RGB/HSV first & second order color moments.

    Returns:
        1D array of length 524.
    """
    # 3D HSV Histogram (8x8x8 = 512 bins)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Masked or global channel statistics
    if mask is not None and np.count_nonzero(mask) > 0:
        indices = mask > 0
        rgb_mean = rgb[indices].mean(axis=0) / 255.0
        rgb_std = rgb[indices].std(axis=0) / 255.0
        hsv_mean = hsv[indices].mean(axis=0) / 255.0
        hsv_std = hsv[indices].std(axis=0) / 255.0
    else:
        rgb_mean = rgb.mean(axis=(0, 1)) / 255.0
        rgb_std = rgb.std(axis=(0, 1)) / 255.0
        hsv_mean = hsv.mean(axis=(0, 1)) / 255.0
        hsv_std = hsv.std(axis=(0, 1)) / 255.0

    return np.concatenate([hist, rgb_mean, rgb_std, hsv_mean, hsv_std]).astype(np.float32)


def extract_texture_features(gray: np.ndarray) -> np.ndarray:
    """Extract GLCM Haralick moments and Uniform Local Binary Pattern (LBP) histogram.

    Returns:
        1D array of length 23 (13 Haralick + 10 LBP).
    """
    # Haralick texture features (13 descriptors averaged over 4 directions)
    haralick = mahotas.features.haralick(gray, ignore_zeros=False).mean(axis=0)

    # LBP (Local Binary Pattern) with P=8, R=1, method='uniform' (10 histogram bins)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    lbp_hist = lbp_hist.astype(np.float32)

    return np.concatenate([haralick, lbp_hist]).astype(np.float32)


def extract_shape_features(gray: np.ndarray, shape_metrics: dict[str, float]) -> np.ndarray:
    """Extract Hu Moments and leaf contour geometry properties.

    Returns:
        1D array of length 13 (7 Hu moments + 6 contour metrics).
    """
    # 7 Hu Moments (scale and rotation invariant, log-transformed)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))

    # 6 segmented leaf contour shape metrics
    contour_props = np.array(
        [
            shape_metrics.get("area_ratio", 1.0),
            shape_metrics.get("perimeter_ratio", 1.0),
            shape_metrics.get("circularity", 0.5),
            shape_metrics.get("aspect_ratio", 1.0),
            shape_metrics.get("extent", 1.0),
            shape_metrics.get("solidity", 1.0),
        ],
        dtype=np.float32,
    )

    return np.concatenate([hu, contour_props]).astype(np.float32)


def extract_gradient_features(gray: np.ndarray) -> np.ndarray:
    """Extract Histogram of Oriented Gradients (HOG) descriptors.

    Uses 128x128 normalized grayscale with 8 orientations and (32, 32) cell size.
    Returns:
        1D array of length 288.
    """
    gray_128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    hog_feats = hog(
        gray_128,
        orientations=8,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )
    return hog_feats.astype(np.float32)


def extract_all_features(image_source: Union[str, Path, np.ndarray]) -> np.ndarray | None:
    """Extract the complete 848-dimensional multimodal feature vector for an image.

    Args:
        image_source: Path to file or BGR NumPy array.

    Returns:
        NumPy float32 array of shape (848,) or None if extraction fails.
    """
    try:
        data = preprocess_image(image_source, target_size=(256, 256), apply_segmentation=True)
    except Exception:
        return None

    try:
        color = extract_color_features(data["hsv"], data["rgb"], data["mask"])
        texture = extract_texture_features(data["gray"])
        shape = extract_shape_features(data["gray"], data["shape_metrics"])
        gradient = extract_gradient_features(data["gray"])

        features = np.concatenate([color, texture, shape, gradient]).astype(np.float32)

        if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
            return None

        return features
    except Exception:
        return None
