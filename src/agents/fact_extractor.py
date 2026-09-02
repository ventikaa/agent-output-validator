from __future__ import annotations

import json

from src.core.llm import get_llm
from src.core.schemas import AgentOutput, Claim

EXTRACTION_PROMPT = """You are a fact extraction agent. Your job is to extract every verifiable factual claim from the given text.

For each claim, identify:
- The specific factual assertion (a number, date, percentage, entity name, or verifiable statement)
- The type: "number", "percentage", "date", "ticker", "entity", or "general"
- The full sentence it was extracted from

Return a JSON array of claims. Only extract concrete, verifiable facts — not opinions, hedging language, or subjective assessments.

Text to analyze:
{content}

Return ONLY valid JSON in this format:
[
  {{"text": "the specific claim", "claim_type": "number", "source_sentence": "the full sentence"}}
]"""


def extract_claims(agent_output: AgentOutput) -> list[Claim]:
    llm = get_llm()
    response = llm.invoke(EXTRACTION_PROMPT.format(content=agent_output.content))

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    parsed = json.loads(raw)
    claims = []
    for i, item in enumerate(parsed):
        claims.append(
            Claim(
                id=i,
                text=item["text"],
                claim_type=item.get("claim_type", "general"),
                source_sentence=item.get("source_sentence", ""),
            )
        )
    return claims
