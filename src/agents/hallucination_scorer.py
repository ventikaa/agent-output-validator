from __future__ import annotations

import json

from src.core.llm import get_llm
from src.core.schemas import FailureCategory, VerificationResult

TAXONOMY_PROMPT = """You are a hallucination taxonomy agent. Given a claim that was found to be contradicted or unsupported, classify the failure into one of these categories:

Categories:
- fabricated_fact: The claim states something that has no basis in any source
- incorrect_number: A numeric value (revenue, percentage, count) is wrong
- misattribution: A fact is attributed to the wrong entity, person, or source
- temporal_error: Dates, time periods, or temporal relationships are wrong
- unsupported_inference: The claim draws a conclusion not supported by the data
- contradicted_by_source: The claim directly contradicts what the source states

Claim: {claim_text}
Verdict: {verdict}
Explanation: {explanation}

Return ONLY valid JSON:
{{"category": "one_of_the_above", "reasoning": "brief reasoning"}}"""


def classify_failure(verification: VerificationResult) -> FailureCategory | None:
    if verification.verdict == "supported":
        return None

    llm = get_llm()
    response = llm.invoke(
        TAXONOMY_PROMPT.format(
            claim_text=verification.claim.text,
            verdict=verification.verdict,
            explanation=verification.explanation,
        )
    )

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    parsed = json.loads(raw)
    category_str = parsed.get("category", "fabricated_fact")

    try:
        return FailureCategory(category_str)
    except ValueError:
        return FailureCategory.FABRICATED_FACT


def score_verifications(verifications: list[VerificationResult]) -> list[VerificationResult]:
    scored = []
    for v in verifications:
        if v.verdict != "supported":
            v.failure_category = classify_failure(v)
        scored.append(v)
    return scored
