"""LangGraph Multi-Agent Orchestrator for AgriGuard-ML.

Connects:
- Agent 1: Image Quality Guard
- Agent 2: Leaf Presence Agent
- Agent 3: Apple Leaf Validator
- Agent 4: Policy Router (RETAKE, REJECT, REVIEW, PASS)
- Agent 5: Severity Estimator
- Agent 6: LightGBM Disease Classifier
- Agent 7: Agricultural Knowledge Retrieval (RAG)
- Agent 8: LLM Advisory & Safety Validation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from src.agents.state import AgriGuardState
from src.agents.agent1_quality import run_quality_agent
from src.agents.agent2_leaf_presence import run_leaf_presence_agent
from src.agents.agent3_apple_validator import run_apple_validator_agent
from src.agents.agent4_policy_router import run_policy_router
from src.agents.agent5_severity import run_severity_agent
from src.agents.agent6_disease_classifier import run_disease_classifier
from src.agents.agent7_rag_retrieval import run_rag_retrieval_agent
from src.agents.agent8_llm_advisory import run_llm_advisory_agent

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


def rejection_terminal_node(state: AgriGuardState) -> AgriGuardState:
    """Terminal node for RETAKE, REJECT, or REVIEW outcomes."""
    decision = state.get("policy_decision", "REJECT")
    reason = state.get("policy_reason", "Validation criteria not satisfied.")
    action = state.get("suggested_action", "Please upload a clearer image.")

    notice = f"""
[AgriGuard-ML: Pipeline Result - {decision}]
* Status: {decision}
* Reason: {reason}
* Next Action: {action}
"""
    return {
        **state,
        "final_advisory": notice,
        "safety_validated": True,
    }


def review_terminal_node(state: AgriGuardState) -> AgriGuardState:
    """Generate explicit expert agronomy review notice when validation or prediction is marginal."""
    reason = state.get("policy_reason", "Verification evidence is marginal or unavailable.")
    action = state.get("suggested_action", "Consult an agricultural extension specialist.")

    notice = f"""
[AgriGuard-ML: Pipeline Result - REVIEW REQUIRED]
* Status: REVIEW (Uncertain Species or Marginal Diagnostic Confidence)
* Reason: {reason}
* Critical Safety Warning: AgriGuard-ML abstains from providing chemical spray guidance when verification evidence is incomplete.
* Recommended Action: {action} Submit physical leaf samples or clear macro photos to your local university extension office.
"""
    return {
        **state,
        "final_advisory": notice,
        "safety_validated": True,
    }


def route_quality_check(state: AgriGuardState) -> str:
    """Short-circuit workflow immediately if image quality failed."""
    if not state.get("quality_passed", False):
        return "policy_router_node"
    return "leaf_presence_node"


def route_leaf_presence_check(state: AgriGuardState) -> str:
    """Short-circuit workflow immediately if no plant foliage was detected."""
    if not state.get("leaf_presence_passed", False):
        return "policy_router_node"
    return "apple_validator_node"


def route_policy_decision(state: AgriGuardState) -> str:
    """Conditional edge router: PASS proceeds to diagnosis, REVIEW routes to expert notice, else rejection."""
    decision = state.get("policy_decision", "REJECT")
    if decision == "PASS":
        return "severity_node"
    if decision == "REVIEW":
        return "review_terminal_node"
    return "rejection_terminal_node"


def build_orchestrator_graph():
    """Build and compile the LangGraph StateGraph with short-circuiting validation gates."""
    if not LANGGRAPH_AVAILABLE:
        return None

    workflow = StateGraph(AgriGuardState)

    # Add modular agent nodes
    workflow.add_node("quality_node", run_quality_agent)
    workflow.add_node("leaf_presence_node", run_leaf_presence_agent)
    workflow.add_node("apple_validator_node", run_apple_validator_agent)
    workflow.add_node("policy_router_node", run_policy_router)

    workflow.add_node("severity_node", run_severity_agent)
    workflow.add_node("disease_classifier_node", run_disease_classifier)
    workflow.add_node("rag_node", run_rag_retrieval_agent)
    workflow.add_node("llm_advisory_node", run_llm_advisory_agent)
    workflow.add_node("rejection_terminal_node", rejection_terminal_node)
    workflow.add_node("review_terminal_node", review_terminal_node)

    # Short-circuiting validation edges
    workflow.set_entry_point("quality_node")
    workflow.add_conditional_edges(
        "quality_node",
        route_quality_check,
        {
            "policy_router_node": "policy_router_node",
            "leaf_presence_node": "leaf_presence_node",
        },
    )
    workflow.add_conditional_edges(
        "leaf_presence_node",
        route_leaf_presence_check,
        {
            "policy_router_node": "policy_router_node",
            "apple_validator_node": "apple_validator_node",
        },
    )
    workflow.add_edge("apple_validator_node", "policy_router_node")

    # Conditional routing after Policy Router
    workflow.add_conditional_edges(
        "policy_router_node",
        route_policy_decision,
        {
            "severity_node": "severity_node",
            "review_terminal_node": "review_terminal_node",
            "rejection_terminal_node": "rejection_terminal_node",
        },
    )

    # Linear diagnosis and advisory edges
    workflow.add_edge("severity_node", "disease_classifier_node")
    workflow.add_edge("disease_classifier_node", "rag_node")
    workflow.add_edge("rag_node", "llm_advisory_node")
    workflow.add_edge("llm_advisory_node", END)
    workflow.add_edge("rejection_terminal_node", END)
    workflow.add_edge("review_terminal_node", END)

    return workflow.compile()


def run_pipeline(image_path: str) -> AgriGuardState:
    """Execute the multi-agent pipeline with LangGraph or short-circuiting sequential fallback."""
    initial_state: AgriGuardState = {
        "image_path": str(image_path),
        "quality_passed": False,
        "leaf_presence_passed": False,
        "apple_validator_passed": False,
        "policy_decision": "PENDING",
        "retrieved_sources": [],
        "execution_trace": ["Pipeline Initiated"],
    }

    graph = build_orchestrator_graph()
    if graph is not None:
        return graph.invoke(initial_state)

    # Short-circuiting sequential fallback
    s = run_quality_agent(initial_state)
    if not s.get("quality_passed", False):
        s = run_policy_router(s)
        return rejection_terminal_node(s)

    s = run_leaf_presence_agent(s)
    if not s.get("leaf_presence_passed", False):
        s = run_policy_router(s)
        return rejection_terminal_node(s)

    s = run_apple_validator_agent(s)
    s = run_policy_router(s)

    dec = s.get("policy_decision")
    if dec == "REJECT" or dec == "RETAKE":
        return rejection_terminal_node(s)
    if dec == "REVIEW":
        return review_terminal_node(s)

    if route_policy_decision(s) == "rejection_terminal_node":
        return rejection_terminal_node(s)

    s = run_severity_agent(s)
    s = run_disease_classifier(s)
    s = run_rag_retrieval_agent(s)
    s = run_llm_advisory_agent(s)
    return s


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "Data/test/Apple___Apple_scab/example.JPG"
    print(f"Executing AgriGuard Multi-Agent LangGraph Pipeline on: {test_file}")
    res = run_pipeline(test_file)
    print("\n" + "=" * 60)
    print("FINAL EXECUTION TRACE:")
    print(" -> ".join(res.get("execution_trace", [])))
    print("=" * 60)
    print(f"Policy Decision: {res.get('policy_decision')}")
    print(f"Disease:         {res.get('disease_name')} ({res.get('disease_confidence', 0)*100:.1f}%)")
    print(f"Severity:        {res.get('severity_percentage')}% ({res.get('severity_tier')})")
    print("\nAdvisory Output:\n", res.get("final_advisory"))
