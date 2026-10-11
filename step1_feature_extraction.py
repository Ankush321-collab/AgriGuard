"""Extract the versioned 848-feature multimodal AppleLeaf9 representation into HDF5.

Includes Leaf Segmentation, LBP & Haralick Texture, Shape Hu Moments & Contour Metrics,
and HOG Gradient Descriptors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.features import FEATURE_COUNT, FEATURE_PIPELINE_VERSION, extract_all_features
from src.feature_pipeline import (
    CLASS_NAMES,
    CLASS_ALIASES,
    IMAGE_EXTENSIONS,
    collect_images,
    run_extraction_pipeline,
)


def image_features(image_path: Path | str) -> np.ndarray | None:
    """Extract full 848-dimensional multimodal feature vector for a single image.

    Backward-compatible entry point for inference scripts.
    """
    return extract_all_features(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 848 multimodal features")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    run_extraction_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()