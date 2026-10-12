"""Phase 6: Comprehensive End-to-End Multi-Agent Workflow Evaluation.

Evaluates:
- Valid Apple leaves across multiple disease classes.
- Other plant / crop leaves (Strawberry, Yucca).
- Defective blurry / dark images.
- Borderline / uncertain cases.
Measures:
- Diagnosis accuracy on accepted images.
- Valid-image false rejections (FRR).
- Invalid-image acceptance (FAR).
- Uncertainty handling & safe abstentions.
- Latency per execution.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.agents.orchestrator import run_pipeline

EVAL_BENCHMARK = [
    # 1. Valid Apple Leaves (Expected: PASS -> Accurate Diagnosis)
    {
        "id": "VAL-01",
        "category": "Valid Apple Foliage",
        "path": "Data/Health/Health (6).jpg",
        "expected_species": "Apple",
        "expected_condition": "Healthy",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-02",
        "category": "Valid Apple Foliage",
        "path": "Data/Health/Health (324).jpg",
        "expected_species": "Apple",
        "expected_condition": "Healthy",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-03",
        "category": "Valid Apple Foliage",
        "path": "Data/Scab/Scab (10).jpg",
        "expected_species": "Apple",
        "expected_condition": "Scab",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-04",
        "category": "Valid Apple Foliage",
        "path": "Data/Rust/Rust (530).jpg",
        "expected_species": "Apple",
        "expected_condition": "Rust",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-05",
        "category": "Valid Apple Foliage",
        "path": "Data/Frogeye leaf spot/Frogeye leaf spot (191).jpg",
        "expected_species": "Apple",
        "expected_condition": "Frogeye leaf spot",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-06",
        "category": "Valid Apple Foliage",
        "path": "Data/Mosaic/Mosaic (163).jpg",
        "expected_species": "Apple",
        "expected_condition": "Mosaic",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-07",
        "category": "Valid Apple Foliage",
        "path": "Data/Powdery mildew/Powdery mildew (1).jpg",
        "expected_species": "Apple",
        "expected_condition": "Powdery mildew",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-08",
        "category": "Valid Apple Foliage",
        "path": "Agentic/tseting/other_fruit_leaf/test3.png",
        "expected_species": "Apple",
        "expected_condition": "Mosaic",
        "expected_decision": "PASS",
    },
    {
        "id": "VAL-09",
        "category": "Valid Apple Foliage",
        "path": "Agentic/tseting/other_fruit_leaf/test5.png",
        "expected_species": "Apple",
        "expected_condition": "Rust",
        "expected_decision": "PASS",
    },

    # 2. Foreign Non-Apple Foliage (Expected: REJECT -> Blocked with Species Citation)
    {
        "id": "NON-01",
        "category": "Non-Apple Foliage",
        "path": "Agentic/tseting/other_fruit_leaf/image.png",
        "expected_species": "Strawberry",
        "expected_condition": "Strawberry Leaf",
        "expected_decision": "REJECT",
    },
    {
        "id": "NON-02",
        "category": "Non-Apple Foliage",
        "path": "Agentic/tseting/other_fruit_leaf/test2.png",
        "expected_species": "Yucca",
        "expected_condition": "Yucca Succulent",
        "expected_decision": "REJECT",
    },
    {
        "id": "NON-03",
        "category": "Non-Apple Foliage",
        "path": "Agentic/tseting/other_fruit_leaf/test6.png",
        "expected_species": "Strawberry",
        "expected_condition": "Strawberry Leaf",
        "expected_decision": "REJECT",
    },

    # 3. Quality Failures / Defective Captures (Expected: RETAKE -> Short-Circuited)
    {
        "id": "DEF-01",
        "category": "Defective Quality",
        "path": "Agentic/tseting/blur/synthetic_heavy_blur.png",
        "expected_species": "Defective",
        "expected_condition": "Heavy Defocus Blur",
        "expected_decision": "RETAKE",
    },
]


def run_phase6_evaluation():
    print("\n" + "=" * 75)
    print("   AGRIGUARD-ML: PHASE 6 END-TO-END WORKFLOW VERIFICATION SUITE")
    print("=" * 75)

    eval_results = []
    total_samples = len(EVAL_BENCHMARK)

    accepted_count = 0
    correct_diagnosis_on_accepted = 0
    valid_apple_total = 0
    valid_apple_accepted = 0
    non_apple_total = 0
    non_apple_rejected = 0
    defective_total = 0
    defective_retaken = 0

    print(f"\n{'ID':<7} {'Category':<20} {'Target File':<30} {'Exp. Dec':<9} {'Act. Dec':<9} {'Latency':<8} {'Status'}")
    print("-" * 95)

    for case in EVAL_BENCHMARK:
        cid = case["id"]
        cat = case["category"]
        p = case["path"]
        exp_dec = case["expected_decision"]
        target_name = Path(p).name

        t0 = time.perf_counter()
        res = run_pipeline(p)
        latency_ms = (time.perf_counter() - t0) * 1000

        act_dec = res.get("policy_decision", "UNKNOWN")
        pred_disease = res.get("disease_name")
        conf = res.get("disease_confidence")
        sources = res.get("retrieved_sources", [])
        trace = res.get("execution_trace", [])

        is_decision_correct = (act_dec == exp_dec)
        status_badge = "SUCCESS" if is_decision_correct else "MISMATCH"

        print(f"{cid:<7} {cat:<20} {target_name:<30} {exp_dec:<9} {act_dec:<9} {latency_ms:>6.0f}ms  {status_badge}")

        # Metrics aggregation
        if cat == "Valid Apple Foliage":
            valid_apple_total += 1
            if act_dec == "PASS":
                valid_apple_accepted += 1
                accepted_count += 1
                if pred_disease and case["expected_condition"].lower() in pred_disease.lower():
                    correct_diagnosis_on_accepted += 1
        elif cat == "Non-Apple Foliage":
            non_apple_total += 1
            if act_dec == "REJECT":
                non_apple_rejected += 1
        elif cat == "Defective Quality":
            defective_total += 1
            if act_dec == "RETAKE":
                defective_retaken += 1

        eval_results.append({
            "test_id": cid,
            "category": cat,
            "file": p,
            "expected_decision": exp_dec,
            "actual_decision": act_dec,
            "expected_condition": case["expected_condition"],
            "predicted_condition": pred_disease,
            "confidence": conf,
            "latency_ms": round(latency_ms, 1),
            "sources_retrieved_count": len(sources),
            "safety_validated": res.get("safety_validated", False),
            "policy_reason": res.get("policy_reason"),
            "is_decision_correct": is_decision_correct,
            "short_circuit_step_count": len(trace),
        })

    # Summary calculations
    valid_acceptance_rate = (valid_apple_accepted / valid_apple_total) * 100 if valid_apple_total else 0
    valid_frr = 100.0 - valid_acceptance_rate
    non_apple_rejection_rate = (non_apple_rejected / non_apple_total) * 100 if non_apple_total else 0
    non_apple_far = 100.0 - non_apple_rejection_rate
    defective_interception_rate = (defective_retaken / defective_total) * 100 if defective_total else 0
    diag_accuracy_on_accepted = (correct_diagnosis_on_accepted / accepted_count) * 100 if accepted_count else 0
    avg_latency = float(np.mean([r["latency_ms"] for r in eval_results]))

    summary = {
        "total_test_cases": total_samples,
        "valid_apple_acceptance_rate_pct": round(valid_acceptance_rate, 2),
        "valid_apple_false_rejection_rate_pct": round(valid_frr, 2),
        "non_apple_rejection_rate_pct": round(non_apple_rejection_rate, 2),
        "non_apple_false_acceptance_rate_pct": round(non_apple_far, 2),
        "defective_quality_interception_rate_pct": round(defective_interception_rate, 2),
        "diagnostic_accuracy_on_accepted_pct": round(diag_accuracy_on_accepted, 2),
        "average_pipeline_latency_ms": round(avg_latency, 1),
    }

    print("\n" + "=" * 75)
    print("   PHASE 6 SCIENTIFIC AUDIT REPORT SUMMARY")
    print("=" * 75)
    print(f"Valid Apple Leaf Acceptance Rate:       {valid_acceptance_rate:.1f}% (FRR: {valid_frr:.1f}%)")
    print(f"Non-Apple Leaf Rejection Rate:          {non_apple_rejection_rate:.1f}% (FAR: {non_apple_far:.1f}%)")
    print(f"Defective Capture Interception Rate:    {defective_interception_rate:.1f}%")
    print(f"Diagnostic Accuracy on Accepted Leaves: {diag_accuracy_on_accepted:.1f}%")
    print(f"Mean Pipeline Execution Latency:        {avg_latency:.1f} ms")
    print("=" * 75)

    # Save to JSON and CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_data = {"summary": summary, "test_cases": eval_results}
    with open(results_dir / "phase6_workflow_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    df = pd.DataFrame(eval_results)
    df.to_csv(results_dir / "phase6_workflow_verification_results.csv", index=False)
    print(f"Full report exported to: {results_dir / 'phase6_workflow_verification_report.json'}")


if __name__ == "__main__":
    run_phase6_evaluation()
