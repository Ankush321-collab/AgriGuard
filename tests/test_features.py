import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.features import (
    FEATURE_COUNT,
    extract_all_features,
    extract_color_features,
    extract_gradient_features,
    extract_shape_features,
    extract_texture_features,
)
from src.preprocess import preprocess_image
from src.segment import segment_leaf


def test_leaf_segmentation_synthetic():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.ellipse(image, (128, 128), (80, 50), 30, 0, 360, (35, 160, 45), -1)

    segmented, mask, shape_metrics = segment_leaf(image)

    assert segmented.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert np.count_nonzero(mask) > 0
    assert "area_ratio" in shape_metrics
    assert "circularity" in shape_metrics
    assert "aspect_ratio" in shape_metrics
    assert "solidity" in shape_metrics


def test_feature_extraction_dimension():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[40:210, 40:210] = [30, 150, 40]

    features = extract_all_features(image)
    assert features is not None
    assert features.shape == (FEATURE_COUNT,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert not np.isnan(features).any()


def test_individual_feature_components():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[50:200, 50:200] = [40, 160, 50]
    pre = preprocess_image(image)

    color_feat = extract_color_features(pre["hsv"], pre["rgb"], pre["mask"])
    assert color_feat.shape == (524,)

    texture_feat = extract_texture_features(pre["gray"])
    assert texture_feat.shape == (23,)

    shape_feat = extract_shape_features(pre["gray"], pre["shape_metrics"])
    assert shape_feat.shape == (13,)

    gradient_feat = extract_gradient_features(pre["gray"])
    assert gradient_feat.shape == (288,)


if __name__ == "__main__":
    test_leaf_segmentation_synthetic()
    print("[PASS] Leaf segmentation test passed.")
    test_feature_extraction_dimension()
    print("[PASS] Full 848-dim feature extraction test passed.")
    test_individual_feature_components()
    print("[PASS] Individual feature families (Color 524, Texture 23, Shape 13, HOG 288) test passed.")
