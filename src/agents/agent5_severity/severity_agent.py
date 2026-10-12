"""Agent 5: Disease Severity Estimation.

Calculates visible lesion coverage percentage:
    visible_percentage = (lesion_area_px / total_leaf_area_px) * 100

Clearly distinguishes visible leaf lesion coverage from total crop yield loss.
Flags segmentation as unreliable when masks are unstable.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.agents.state import AgriGuardState


def run_severity_agent(state: AgriGuardState) -> AgriGuardState:
    """Evaluate disease severity against verified reference masks.
    
    Phase 4 Requirement: Withhold quantitative severity percentage as unavailable
    until validated against expert-annotated reference benchmark masks.
    """
    trace = state.get("execution_trace", []) + ["Agent 5: Severity Estimation"]

    return {
        **state,
        "severity_percentage": None,
        "severity_tier": "Unavailable (No benchmark reference masks)",
        "segmentation_reliable": False,
        "severity_status": "UNAVAILABLE",
        "severity_note": (
            "Severity quantification is withheld: Pixel-level lesion masks have not undergone "
            "reference Dice/IoU benchmark validation on expert-annotated ground truth masks."
        ),
        "execution_trace": trace,
    }
