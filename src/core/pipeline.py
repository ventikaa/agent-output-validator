from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.citation_verifier import CitationVerifier
from src.agents.escalation_router import build_report
from src.agents.fact_extractor import extract_claims
from src.agents.hallucination_scorer import score_verifications
from src.core.indexer import CorpusIndexer
from src.core.schemas import AgentOutput, Claim, ConfidenceReport, VerificationResult


class ValidatorState(TypedDict):
    agent_output: AgentOutput
    claims: list[Claim]
    verifications: list[VerificationResult]
    report: ConfidenceReport | None


def build_pipeline(corpus_dir: str = "data/corpus") -> tuple[StateGraph, CorpusIndexer]:
    indexer = CorpusIndexer(corpus_dir=corpus_dir)
    indexer.build_index()
    verifier = CitationVerifier(indexer)

    def extract_node(state: ValidatorState) -> dict:
        claims = extract_claims(state["agent_output"])
        return {"claims": claims}

    def verify_node(state: ValidatorState) -> dict:
        verifications = verifier.verify_all(state["claims"])
        return {"verifications": verifications}

    def score_node(state: ValidatorState) -> dict:
        scored = score_verifications(state["verifications"])
        return {"verifications": scored}

    def route_node(state: ValidatorState) -> dict:
        report = build_report(
            agent_output_id=state["agent_output"].id,
            verifications=state["verifications"],
        )
        return {"report": report}

    graph = StateGraph(ValidatorState)
    graph.add_node("extract", extract_node)
    graph.add_node("verify", verify_node)
    graph.add_node("score", score_node)
    graph.add_node("route", route_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", "score")
    graph.add_edge("score", "route")
    graph.add_edge("route", END)

    return graph, indexer


def validate_output(agent_output: AgentOutput, corpus_dir: str = "data/corpus") -> ConfidenceReport:
    graph, _ = build_pipeline(corpus_dir)
    app = graph.compile()

    initial_state: ValidatorState = {
        "agent_output": agent_output,
        "claims": [],
        "verifications": [],
        "report": None,
    }

    result = app.invoke(initial_state)
    return result["report"]
