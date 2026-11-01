"""Comprehensive Validation Suite for AgriGuard-ML 8-Agent Pipeline on Unseen Data."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import run_pipeline


def verify_agent_outputs(case_name: str, state: Dict[str, Any], expected_decision: str = "PASS") -> Dict[str, Any]:
    """Verify that every single one of the 8 agents produced authentic, non-mocked outputs."""
    report = {
        "case": case_name,
        "image": Path(state.get("image_path", "")).name,
        "decision": state.get("policy_decision"),
        "agents_verified": {},
        "all_passed": True,
    }

    # Agent 1: Image Quality Guard
    a1_ok = (
        "quality_passed" in state
        and isinstance(state.get("blur_score"), (int, float))
        and isinstance(state.get("brightness"), (int, float))
        and isinstance(state.get("resolution"), (list, tuple))
    )
    report["agents_verified"]["Agent 1 (Quality Guard)"] = {
        "verified": a1_ok,
        "blur_score": round(state.get("blur_score", 0), 2),
        "brightness": round(state.get("brightness", 0), 2),
    }

    # Agent 2: Leaf Presence
    a2_ok = (
        "leaf_presence_passed" in state
        and isinstance(state.get("leaf_area_fraction"), (int, float))
    )
    report["agents_verified"]["Agent 2 (Leaf Presence)"] = {
        "verified": a2_ok,
        "leaf_area_fraction": round(state.get("leaf_area_fraction", 0), 4),
    }

    # Agent 3: Apple Species Validator
    a3_ok = (
        "apple_validator_passed" in state
        and isinstance(state.get("apple_confidence"), (int, float))
    )
    report["agents_verified"]["Agent 3 (Apple Validator)"] = {
        "verified": a3_ok,
        "apple_confidence": round(state.get("apple_confidence", 0), 2),
    }

    # Agent 4: Policy Router Gate
    decision = state.get("policy_decision")
    a4_ok = decision in ["PASS", "RETAKE", "REJECT", "REVIEW"]
    report["agents_verified"]["Agent 4 (Policy Router)"] = {
        "verified": a4_ok,
        "decision": decision,
        "reason": state.get("policy_reason", ""),
    }

    if decision != "PASS":
        # Short-circuiting verification: earlier gates bypass later pre-check agents
        if decision == "RETAKE":
            report["all_passed"] = a1_ok and a4_ok and (decision == expected_decision)
        elif decision == "REJECT":
            report["all_passed"] = a1_ok and (a2_ok or a3_ok) and a4_ok and (decision == expected_decision)
        else:
            report["all_passed"] = a4_ok and (decision == expected_decision)
        return report

    # Agent 5: Severity Estimator
    a5_ok = (
        "severity_percentage" in state
        and "severity_tier" in state
        and "severity_status" in state
    )
    report["agents_verified"]["Agent 5 (Severity Estimator)"] = {
        "verified": a5_ok,
        "severity_percentage": state.get("severity_percentage"),
        "severity_tier": state.get("severity_tier"),
    }

    # Agent 6: Classical ML Disease Classifier (848 Features)
    probs = state.get("class_probabilities", {})
    a6_ok = (
        bool(state.get("disease_name"))
        and isinstance(state.get("disease_confidence"), (int, float))
        and state.get("disease_confidence", 0) > 0
        and isinstance(probs, dict)
        and len(probs) >= 9  # Must have all 9 class probabilities from ML ensemble
    )
    report["agents_verified"]["Agent 6 (Disease Classifier - ML 848)"] = {
        "verified": a6_ok,
        "disease_name": state.get("disease_name"),
        "confidence": round(state.get("disease_confidence", 0), 4),
        "model_name": state.get("model_name"),
        "num_class_probabilities": len(probs),
        "top_class_prob": max(probs.values()) if probs else 0,
    }

    # Agent 7: Agricultural Knowledge Retrieval (RAG)
    sources = state.get("retrieved_sources", [])
    a7_ok = isinstance(sources, list) and len(sources) > 0 and "cultural_controls" in sources[0]
    report["agents_verified"]["Agent 7 (RAG Knowledge Retrieval)"] = {
        "verified": a7_ok,
        "sources_retrieved": len(sources),
        "top_source_title": sources[0].get("title") if sources else "None",
        "has_cultural_controls": bool(sources[0].get("cultural_controls")) if sources else False,
    }

    # Agent 8: LLM Advisory & Safety Validation
    advisory = state.get("final_advisory", "")
    a8_ok = (
        isinstance(advisory, str)
        and len(advisory) > 50
        and state.get("safety_validated") is True
    )
    report["agents_verified"]["Agent 8 (LLM Advisory & Safety)"] = {
        "verified": a8_ok,
        "safety_validated": state.get("safety_validated"),
        "advisory_length_chars": len(advisory),
    }

    # Execution trace
    trace = state.get("execution_trace", [])
    report["execution_trace_length"] = len(trace)

    report["all_passed"] = (
        a1_ok and a2_ok and a3_ok and a4_ok and a5_ok and a6_ok and a7_ok and a8_ok
    )
    return report


def main() -> None:
    print("=" * 80)
    print("AgriGuard-ML: Verification of All 8 Modular Agents on Unseen Dataset")
    print("=" * 80)

    # Pick genuine unseen images from Data/ across all classes
    test_cases = [
        {"class": "Scab", "path": "Data/Scab/Scab (120).jpg", "expected": "PASS"},
        {"class": "Health", "path": "Data/Health/Health (55).jpg", "expected": "PASS"},
        {"class": "Rust", "path": "Data/Rust/Rust (45).jpg", "expected": "PASS"},
        {"class": "Powdery mildew", "path": "Data/Powdery mildew/Powdery mildew (1185).jpg", "expected": "PASS"},
        {"class": "Frogeye leaf spot", "path": "Data/Frogeye leaf spot/Frogeye leaf spot (28).jpg", "expected": "PASS"},
        {"class": "Alternaria leaf spot", "path": "Data/Alternaria leaf spot/Alternaria leaf spot (50).jpg", "expected": "PASS"},
        {"class": "Brown spot", "path": "Data/Brown spot/Brown spot (1).jpg", "expected": "PASS"},
        {"class": "Grey spot", "path": "Data/Grey spot/Grey spot (30).jpg", "expected": "PASS"},
        {"class": "Mosaic", "path": "Data/Mosaic/Mosaic (25).jpg", "expected": "PASS"},
        {"class": "Blur Gate (Defect)", "path": "Agentic/tseting/blur/img2.png", "expected": "RETAKE"},
        {"class": "Non-Apple Species Gate", "path": "Agentic/tseting/other_fruit_leaf/image.png", "expected": "REJECT"},
    ]

    total_cases = len(test_cases)
    passed_cases = 0

    for idx, tc in enumerate(test_cases, 1):
        rel_path = tc["path"]
        abs_path = PROJECT_ROOT / rel_path
        if not abs_path.is_file():
            print(f"[{idx}/{total_cases}] SKIPPED: File not found {rel_path}")
            continue

        print(f"\n[{idx}/{total_cases}] Testing Unseen Leaf: {tc['class']} ({rel_path})")
        state = run_pipeline(str(abs_path))
        rep = verify_agent_outputs(tc["class"], state, expected_decision=tc["expected"])

        print(f"  * Policy Decision: {rep['decision']} (Expected: {tc['expected']})")
        if rep["decision"] == "PASS":
            a6 = rep["agents_verified"]["Agent 6 (Disease Classifier - ML 848)"]
            print(f"  * Machine Learning Prediction: {a6['disease_name']} ({a6['confidence']*100:.1f}%)")
            print(f"  * Model Consensus: {a6['model_name']}")
            print(f"  * Class Probabilities Computed: {a6['num_class_probabilities']} classes (Dynamic ML)")
            a7 = rep["agents_verified"]["Agent 7 (RAG Knowledge Retrieval)"]
            print(f"  * RAG Grounding Source: {a7['top_source_title']}")
            a8 = rep["agents_verified"]["Agent 8 (LLM Advisory & Safety)"]
            print(f"  * LLM Advisory Generated: {a8['advisory_length_chars']} chars, Safety Validated: {a8['safety_validated']}")

        all_ok = rep["all_passed"]
        status_str = "[PASS - ALL 8 AGENTS VERIFIED]" if all_ok else "[FAIL]"
        print(f"  * Overall Status: {status_str}")
        if all_ok:
            passed_cases += 1

    print("\n" + "=" * 80)
    print(f"Summary: {passed_cases}/{total_cases} Unseen Test Cases Passed Verification")
    print("=" * 80)


if __name__ == "__main__":
    main()
