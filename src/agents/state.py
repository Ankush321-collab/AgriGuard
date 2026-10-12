"""Shared state definition for the AgriGuard-ML LangGraph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import numpy as np


class AgriGuardState(TypedDict, total=False):
    """Unified state schema passed across all modular agents."""

    # Input Information
    image_path: str
    image_bgr: Optional[np.ndarray]

    # Agent 1: Image Quality
    quality_passed: bool
    blur_score: float
    brightness: float
    resolution: List[int]
    quality_reason: Optional[str]

    # Agent 2: Leaf Presence
    leaf_presence_passed: bool
    leaf_area_fraction: float
    leaf_mask: Optional[np.ndarray]
    leaf_presence_reason: Optional[str]

    # Agent 3: Apple Leaf Validator
    apple_validator_passed: bool
    apple_confidence: float
    is_apple_leaf: bool
    apple_validator_reason: Optional[str]

    # Agent 4: Policy Router
    policy_decision: str  # "PASS", "RETAKE", "REJECT", "REVIEW"
    policy_reason: Optional[str]
    suggested_action: Optional[str]

    # Agent 5: Severity Estimation
    severity_percentage: Optional[float]
    severity_tier: Optional[str]
    lesion_mask: Optional[np.ndarray]
    segmentation_reliable: bool

    # Agent 6: Disease Classification (LightGBM Core)
    disease_name: Optional[str]
    disease_confidence: Optional[float]
    model_name: Optional[str]
    is_uncertain_prediction: bool

    # Agent 7: Agricultural Knowledge Retrieval (RAG)
    retrieved_sources: List[Dict[str, Any]]
    evidence_coverage: float

    # Agent 8: LLM Advisory & Safety Validation
    final_advisory: Optional[str]
    safety_validated: bool
    execution_trace: List[str]
    error_message: Optional[str]
