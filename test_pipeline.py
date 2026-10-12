"""AgriGuard-ML: Quick Pipeline Verification & Testing Script.

Usage:
1. Open this file and paste your image path into IMAGE_PATH below.
2. Run in terminal:
       python test_pipeline.py
   or
       python test_pipeline.py "path/to/your/image.jpg"
"""

from __future__ import annotations

import sys
from pathlib import Path

# ==============================================================================
# PASTE YOUR IMAGE PATH HERE TO TEST:
# ==============================================================================
# IMAGE_PATH = "Agentic/tseting/other_fruit_leaf/test2.png"
IMAGE_PATH = "Agentic/tseting/blur/img2.png"


# Quick Sample Test Paths (uncomment any to test):
# IMAGE_PATH = "Data/Health/Health (6).jpg"                      # Valid Healthy Apple Leaf (PASS)
# IMAGE_PATH = "Data/Scab/Scab (10).jpg"                         # Valid Apple Scab (PASS)
# IMAGE_PATH = "Agentic/tseting/blur/blur1.png"                 # Valid Apple Rust (PASS)
# IMAGE_PATH = "Agentic/tseting/blur/synthetic_heavy_blur.png"   # Blurry Image (RETAKE)
# IMAGE_PATH = "Agentic/tseting/other_fruit_leaf/image.png"     # Strawberry Leaf (REJECT)
# ==============================================================================


def main() -> None:
    # Allow command-line override if an argument is passed
    target_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    img_file = Path(target_path)

    print("\n" + "=" * 70)
    print("   AGRIGUARD-ML: MULTI-AGENT PIPELINE VERIFICATION RUNNER")
    print("=" * 70)
    print(f"Target Image: {img_file.resolve()}")

    if not img_file.is_file():
        print(f"\n[ERROR] File not found: '{img_file}'")
        print("Please check the path and try again.")
        return

    # Import orchestrator runner with clear environment error handling
    try:
        from src.agents.orchestrator import run_pipeline
    except ModuleNotFoundError as err:
        if "cv2" in str(err):
            print("\n" + "!" * 70)
            print("[ENVIRONMENT ERROR] 'cv2' (OpenCV) was not found in this Python interpreter:")
            print(f"  Current Python: {sys.executable}")
            print("\nYou ran with the global Python instead of the project virtual environment.")
            print("To fix, simply run with the virtual environment:")
            print("   .\\.venv\\Scripts\\python.exe test_pipeline.py")
            print("or activate the virtual environment and run:")
            print("   python test_pipeline.py")
            print("!" * 70 + "\n")
            return
        raise

    print("\nExecuting Multi-Agent Workflow...")
    result = run_pipeline(str(img_file))

    # Format Output Report
    decision = result.get("policy_decision", "UNKNOWN")
    decision_badge = {
        "PASS": "[PASS] VALIDATED APPLE LEAF -> DIAGNOSED",
        "RETAKE": "[RETAKE] QUALITY ISSUE -> RETAKE ADVICE GIVEN",
        "REJECT": "[REJECT] OUTSIDE SCOPE -> NON-APPLE LEAF BLOCKED",
        "REVIEW": "[REVIEW] MARGINAL CONFIDENCE -> FLAGGED FOR REVIEW",
    }.get(decision, f"[{decision}]")

    print("\n" + "-" * 70)
    print(f"PIPELINE STATUS: {decision_badge}")
    print("-" * 70)

    # Execution Trace
    trace = result.get("execution_trace", [])
    if trace:
        print("\nAgent Execution Sequence:")
        for idx, step in enumerate(trace, 1):
            print(f"  {idx}. {step}")

    # Stage 2 Validation Details
    print("\nValidation Gate Diagnostics:")
    print(f"  • Quality Check:     {'PASSED' if result.get('quality_passed') else 'FAILED'}")
    if result.get("blur_score") is not None:
        print(f"    - Focus Score:     {result.get('blur_score'):.1f} (Threshold >= 8.0)")
        print(f"    - Brightness:      {result.get('brightness'):.1f} (Range: 35 - 225)")
    print(f"  • Leaf Presence:     {'CONFIRMED' if result.get('leaf_presence_passed') else 'ABSENT'}")
    print(f"  • Apple Species:     {'VERIFIED' if result.get('apple_validator_passed') else 'REJECTED'}")
    if result.get("apple_validator_reason"):
        print(f"    - Reason:          {result.get('apple_validator_reason')}")

    # Stage 1 ML Core Details (if passed)
    if decision in ("PASS", "REVIEW") and result.get("disease_name"):
        print("\nDisease Diagnostic Core:")
        print(f"  • Predicted Disease: {result.get('disease_name')}")
        conf = result.get("disease_confidence", 0.0)
        print(f"  • Model Confidence:  {conf * 100:.1f}%")
        print(f"  • Model Used:        {result.get('model_name', 'LightGBM')}")
        print(f"  • Visible Severity:  {result.get('severity_percentage')}% ({result.get('severity_tier')})")

    # Final Clinical Advisory
    print("\n" + "=" * 70)
    print("FINAL ADVISORY / RECOMMENDATION OUTPUT:")
    print("=" * 70)
    print(result.get("final_advisory", "No advisory generated."))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
