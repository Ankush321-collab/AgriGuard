"""Run feature extraction followed by four-model evaluation."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "step1_feature_extraction.py"], check=True)
    subprocess.run([sys.executable, "step2_train_4_models.py", "--device", "auto"], check=True)
    print("Pipeline complete. See embeddings/, models/, and results/.")


if __name__ == "__main__":
    main()