from __future__ import annotations

from collections import Counter

from src.core.schemas import ConfidenceReport, EscalationLevel, VerificationResult


def build_report(
    agent_output_id: str,
    verifications: list[VerificationResult],
) -> ConfidenceReport:
    total = len(verifications)
    if total == 0:
        return ConfidenceReport(
            agent_output_id=agent_output_id,
            overall_confidence=1.0,
            escalation=EscalationLevel.PASS,
            total_claims=0,
            supported_count=0,
            contradicted_count=0,
            unsupported_count=0,
            verifications=[],
            summary="No verifiable claims found in agent output.",
        )

    supported = sum(1 for v in verifications if v.verdict == "supported")
    contradicted = sum(1 for v in verifications if v.verdict == "contradicted")
    unsupported = sum(1 for v in verifications if v.verdict == "unsupported")

    confidence_scores = [v.confidence for v in verifications]
    overall = sum(confidence_scores) / len(confidence_scores)

    failure_cats = Counter()
    for v in verifications:
        if v.failure_category:
            failure_cats[v.failure_category.value] += 1

    if overall >= 0.8 and contradicted == 0:
        escalation = EscalationLevel.PASS
    elif overall < 0.5 or contradicted >= total * 0.5:
        escalation = EscalationLevel.BLOCK
    else:
        escalation = EscalationLevel.FLAG

    summary_parts = [f"{total} claims analyzed."]
    if supported:
        summary_parts.append(f"{supported} supported.")
    if contradicted:
        summary_parts.append(f"{contradicted} contradicted.")
    if unsupported:
        summary_parts.append(f"{unsupported} unsupported.")
    summary_parts.append(f"Overall confidence: {overall:.1%}.")
    summary_parts.append(f"Decision: {escalation.value.upper()}.")

    return ConfidenceReport(
        agent_output_id=agent_output_id,
        overall_confidence=round(overall, 4),
        escalation=escalation,
        total_claims=total,
        supported_count=supported,
        contradicted_count=contradicted,
        unsupported_count=unsupported,
        failure_breakdown=dict(failure_cats),
        verifications=verifications,
        summary=" ".join(summary_parts),
    )
