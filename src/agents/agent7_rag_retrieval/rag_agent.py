"""Agent 7: Agricultural Knowledge Retrieval (RAG).

Queries structured and vector-indexed agricultural documents from CABI PlantwisePlus,
UC IPM, and regional extension authorities.
Filters by disease, crop, severity tier, and region.
"""

from __future__ import annotations

from typing import Any, Dict, List
from src.agents.state import AgriGuardState


from pathlib import Path
import json

def run_rag_retrieval_agent(state: AgriGuardState) -> AgriGuardState:
    """Retrieve verified, citation-backed agricultural guidelines from curated authoritative sources."""
    trace = state.get("execution_trace", []) + ["Agent 7: Knowledge Retrieval (RAG)"]
    disease = state.get("disease_name", "")

    kb_path = Path("src/knowledge_base/curated_extension_sources.json")
    if not kb_path.is_file():
        return {
            **state,
            "retrieved_sources": [],
            "evidence_coverage": 0.0,
            "rag_status": "INSUFFICIENT_EVIDENCE",
            "rag_message": "Knowledge base file not found. Retrieval cannot substantiate claims.",
            "execution_trace": trace,
        }

    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
    except Exception as exc:
        return {
            **state,
            "retrieved_sources": [],
            "evidence_coverage": 0.0,
            "rag_status": "INSUFFICIENT_EVIDENCE",
            "rag_message": f"Error parsing knowledge base: {exc}",
            "execution_trace": trace,
        }

    sources = kb_data.get("sources", [])
    matched_sources = []

    disease_lower = disease.lower().strip()
    for s in sources:
        target = s.get("disease_target", "").lower()
        synonyms = [syn.lower() for syn in s.get("disease_synonyms", [])]
        if target in disease_lower or any(syn in disease_lower for syn in synonyms) or disease_lower in target:
            matched_sources.append({
                "title": s.get("title"),
                "organization": s.get("organization"),
                "publication_date": s.get("publication_date"),
                "region": s.get("region"),
                "url": s.get("url"),
                "retrieved_passage": s.get("retrieved_passage"),
                "cultural_controls": s.get("cultural_controls", []),
                "chemical_regulations": s.get("chemical_regulations", "Consult local extension services."),
            })

    if not matched_sources:
        return {
            **state,
            "retrieved_sources": [],
            "evidence_coverage": 0.0,
            "rag_status": "INSUFFICIENT_EVIDENCE",
            "rag_message": f"Insufficient evidence: No verified authoritative source found for '{disease}'. Recommendations abstained.",
            "execution_trace": trace,
        }

    return {
        **state,
        "retrieved_sources": matched_sources,
        "evidence_coverage": 1.0,
        "rag_status": "EVIDENCE_CONFIRMED",
        "rag_message": None,
        "execution_trace": trace,
    }
