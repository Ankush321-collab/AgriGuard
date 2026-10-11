"""Image preprocessing module for AgriGuard-ML.

Standardizes images to 256x256, converts between RGB, HSV, and Grayscale,
and interfaces with the leaf segmentation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import cv2
import numpy as np

from src.segment import segment_leaf


def load_image(image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from a file path or validate an existing NumPy array."""
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at path: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image from path: {path}")
        return image
    elif isinstance(image_source, np.ndarray):
        if image_source.size == 0 or len(image_source.shape) < 2:
            raise ValueError("Invalid numpy array passed as image.")
        return image_source
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")


def preprocess_image(
    image_source: Union[str, Path, np.ndarray],
    target_size: tuple[int, int] = (256, 256),
    apply_segmentation: bool = True,
) -> dict[str, Union[np.ndarray, dict[str, float]]]:
    """Standardize an image and generate color spaces, leaf masks, and shape metrics.

    Args:
        image_source: Path to file or BGR NumPy array.
        target_size: Output image dimensions (width, height).
        apply_segmentation: Whether to perform leaf segmentation.

    Returns:
        dict containing:
            - 'bgr': Resized BGR image.
            - 'rgb': Standardized RGB image.
            - 'hsv': HSV color space image.
            - 'gray': Grayscale image.
            - 'mask': Binary leaf mask.
            - 'segmented_bgr': Masked leaf BGR image.
            - 'shape_metrics': Extracted geometric shape features.
    """
    image = load_image(image_source)
    resized_bgr = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    if apply_segmentation:
        segmented_bgr, mask, shape_metrics = segment_leaf(resized_bgr, target_size=target_size)
    else:
        segmented_bgr = resized_bgr
        mask = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
        h, w = target_size[1], target_size[0]
        shape_metrics = {
            "area": float(h * w),
            "perimeter": float(2 * (h + w)),
            "area_ratio": 1.0,
            "perimeter_ratio": 1.0,
            "circularity": 0.5,
            "aspect_ratio": 1.0,
            "extent": 1.0,
            "solidity": 1.0,
        }

    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "bgr": resized_bgr,
        "rgb": rgb,
        "hsv": hsv,
        "gray": gray,
        "mask": mask,
        "segmented_bgr": segmented_bgr,
        "shape_metrics": shape_metrics,
    }
