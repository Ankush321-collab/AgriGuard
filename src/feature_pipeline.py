"""Feature extraction pipeline for AgriGuard-ML.

Extracts the full 848-dimensional multimodal feature representation
(Color, Texture [Haralick + LBP], Shape [Hu + Contour], Gradient [HOG])
across the AppleLeaf9 dataset into versioned HDF5 storage using parallel processing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from tqdm import tqdm

from src.features import FEATURE_COUNT, FEATURE_PIPELINE_VERSION, extract_all_features

CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Frogeye leaf spot",
    "Grey spot",
    "Healthy",
    "Mosaic",
    "Powdery mildew",
    "Rust",
    "Scab",
]
CLASS_ALIASES = {"Health": "Healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def collect_images(data_dir: Path) -> list[tuple[Path, int]]:
    """Scan the dataset directory and map images to their integer class label."""
    class_directories = {}
    for directory in data_dir.iterdir():
        if not directory.is_dir() or directory.name == "outputs":
            continue
        canonical_name = CLASS_ALIASES.get(directory.name, directory.name)
        if canonical_name in CLASS_NAMES:
            class_directories[canonical_name] = directory

    missing = [name for name in CLASS_NAMES if name not in class_directories]
    if missing:
        raise FileNotFoundError(f"Missing class folders in {data_dir}: {', '.join(missing)}")

    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        for path in sorted(class_directories[class_name].rglob("*")):
            if path.is_file() and path.suffix in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


def _worker_extract(item: tuple[Path, int]) -> tuple[np.ndarray | None, int, str]:
    """Worker function for multiprocessing."""
    path, label = item
    feats = extract_all_features(path)
    return feats, label, str(path)


def run_extraction_pipeline(
    data_dir: Path = Path("Data"),
    output_dir: Path = Path("embeddings"),
    max_workers: int | None = None,
) -> tuple[int, int]:
    """Execute feature extraction on all images and save to HDF5.

    Returns:
        tuple (processed_count, skipped_count)
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 12)

    samples = collect_images(data_dir)
    print(f"Discovered {len(samples)} images across {len(CLASS_NAMES)} classes.")
    print(f"Extracting features using {max_workers} worker processes...")

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    skipped = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(_worker_extract, samples, chunksize=32),
                total=len(samples),
                desc="Extracting 848-dim features",
            )
        )

    for feats, label, path_str in results:
        if feats is None:
            skipped += 1
            continue
        features.append(feats)
        labels.append(label)
        paths.append(path_str)

    if not features:
        raise RuntimeError("No readable images produced valid features.")

    output_dir.joinpath("features").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("labels").mkdir(parents=True, exist_ok=True)

    feature_matrix = np.stack(features)
    label_array = np.asarray(labels, dtype=np.int64)
    path_array = np.asarray(paths, dtype="S")

    features_file = output_dir / "features" / "features.h5"
    with h5py.File(features_file, "w") as file:
        file.create_dataset("features", data=feature_matrix, compression="gzip")
        file.create_dataset("paths", data=path_array, compression="gzip")
        file.attrs["feature_count"] = FEATURE_COUNT
        file.attrs["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION
        file.attrs["resize"] = "256x256"
        file.attrs["segmentation"] = "exg-hsv-morphology-contour"
        file.attrs["color_features"] = "hsv_hist_512+rgb_moments_6+hsv_moments_6"
        file.attrs["texture_features"] = "haralick_13+lbp_uniform_10"
        file.attrs["shape_features"] = "hu_moments_7+leaf_contour_props_6"
        file.attrs["gradient_features"] = "hog_288"

    labels_file = output_dir / "labels" / "labels.h5"
    with h5py.File(labels_file, "w") as file:
        file.create_dataset("labels", data=label_array, compression="gzip")
        file.create_dataset("class_names", data=np.asarray(CLASS_NAMES, dtype="S"))
        file.attrs["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION

    print(f"\nFeature extraction complete:")
    print(f"  Processed images: {len(feature_matrix)}")
    print(f"  Skipped images:   {skipped}")
    print(f"  Feature shape:    {feature_matrix.shape}")
    print(f"  Saved HDF5:       {features_file} & {labels_file}")
    return len(feature_matrix), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="AgriGuard-ML Feature Pipeline")
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
