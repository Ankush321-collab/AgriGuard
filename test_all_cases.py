"""Comprehensive 50-Case Verification Suite for AgriGuard-ML Multi-Agent Pipeline."""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.orchestrator import run_pipeline

CATEGORIES = [
    ("Health", "Healthy"),
    ("Scab", "Scab"),
    ("Rust", "Rust"),
    ("Powdery mildew", "Powdery mildew"),
    ("Frogeye leaf spot", "Frogeye leaf spot"),
    ("Alternaria leaf spot", "Alternaria leaf spot"),
    ("Brown spot", "Brown spot"),
    ("Grey spot", "Grey spot"),
    ("Mosaic", "Mosaic"),
]

def main():
    print("=" * 80)
    print("      AGRIGUARD-ML: COMPREHENSIVE 50 TEST CASES VERIFICATION SUITE")
    print("=" * 80)

    total_disease_cases = 0
    correct_disease_cases = 0

    results = []

    # 1. Test 5 samples for each of the 9 AppleLeaf9 categories (45 cases)
    for folder_name, expected_class in CATEGORIES:
        files = sorted(glob.glob(f"Data/{folder_name}/*.*"))[:5]
        for f_path in files:
            p = Path(f_path)
            t0 = time.time()
            res = run_pipeline(str(p))
            latency = (time.time() - t0) * 1000

            pred = res.get("disease_name")
            conf = res.get("disease_confidence", 0.0)
            decision = res.get("policy_decision")
            passed = res.get("quality_passed", False)
            focus = res.get("blur_score", 0.0)

            is_correct = (pred == expected_class)
            total_disease_cases += 1
            if is_correct:
                correct_disease_cases += 1

            results.append({
                "type": "disease",
                "file": p.name,
                "category": expected_class,
                "expected": expected_class,
                "predicted": pred,
                "confidence": conf,
                "decision": decision,
                "focus": focus,
                "is_correct": is_correct,
                "latency_ms": latency,
            })

    # 2. Test Blur Cases (3 cases)
    blur_files = [
        ("Agentic/tseting/blur/img2.png", "RETAKE", "Blurry Image"),
        ("Agentic/tseting/blur/blur1.png", "PASS", "Sharp Specimen"),
    ]
    for b_path, exp_decision, desc in blur_files:
        if os.path.exists(b_path):
            p = Path(b_path)
            res = run_pipeline(str(p))
            decision = res.get("policy_decision")
            focus = res.get("blur_score", 0.0)
            is_ok = (decision == exp_decision)
            results.append({
                "type": "blur",
                "file": p.name,
                "category": "Blur Check",
                "expected": exp_decision,
                "predicted": decision,
                "confidence": focus,
                "decision": decision,
                "focus": focus,
                "is_correct": is_ok,
                "desc": desc,
            })

    # 3. Test Out-of-Scope / Non-Apple Cases
    non_apple_files = [
        ("Agentic/tseting/other_fruit_leaf/image.png", "REJECT", "Iris/Strawberry"),
        ("Agentic/tseting/other_fruit_leaf/test2.png", "REJECT", "Yucca/Non-Apple"),
        ("Agentic/tseting/other_fruit_leaf/orange_dis.png", "REJECT", "Orange/Citrus"),
        ("Agentic/tseting/other_fruit_leaf/pinapple_dis.png", "REJECT", "Pineapple"),
    ]
    for na_path, exp_decision, desc in non_apple_files:
        if os.path.exists(na_path):
            p = Path(na_path)
            res = run_pipeline(str(p))
            decision = res.get("policy_decision")
            is_ok = (decision == exp_decision)
            results.append({
                "type": "out_of_scope",
                "file": p.name,
                "category": "Species Gate",
                "expected": exp_decision,
                "predicted": decision,
                "confidence": res.get("apple_confidence", 0.0),
                "decision": decision,
                "focus": res.get("blur_score", 0.0),
                "is_correct": is_ok,
                "desc": desc,
            })

    # Print Summary Table
    print(f"\nCompleted {len(results)} total test cases:")
    print(f"Disease Classification Accuracy: {correct_disease_cases}/{total_disease_cases} ({correct_disease_cases/total_disease_cases*100:.1f}%)")
    print("-" * 80)
    print(f"{'Category':<22} | {'File':<16} | {'Decision':<8} | {'Focus':<7} | {'Prediction':<22} | {'Conf':<6} | {'Status'}")
    print("-" * 80)

    for r in results:
        status_str = "PASS [OK]" if r["is_correct"] else "FAIL [MISMATCH]"
        if r["type"] == "disease":
            print(f"{r['category']:<22} | {r['file'][:15]:<16} | {r['decision']:<8} | {r['focus']:<7.1f} | {str(r['predicted']):<22} | {r['confidence']*100:<5.1f}% | {status_str}")
        else:
            print(f"{r['category']:<22} | {r['file'][:15]:<16} | {r['decision']:<8} | {r['focus']:<7.1f} | Gate: {r['expected']:<16} | {r['decision']:<6} | {status_str}")

    print("=" * 80)

if __name__ == "__main__":
    main()
