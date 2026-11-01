"""AgriGuard-ML JSON Pipeline Runner for Next.js Frontend Integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.agents.orchestrator import run_pipeline


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects for JSON output."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        # Don't dump multi-megabyte image arrays into JSON; omit or summarize
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if not k.endswith("_bgr") and not k.endswith("_mask")}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).is_file():
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    try:
        state = run_pipeline(image_path)
        clean_state = sanitize_for_json(state)
        # Output pure JSON marker for robust parsing
        print("__AGRIGUARD_JSON_START__")
        print(json.dumps(clean_state, indent=2))
        print("__AGRIGUARD_JSON_END__")
    except Exception as exc:
        print(json.dumps({"error": str(exc), "trace": [str(exc)]}))
        sys.exit(1)


if __name__ == "__main__":
    main()
