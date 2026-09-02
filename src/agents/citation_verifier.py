from __future__ import annotations

import json

from src.core.indexer import CorpusIndexer
from src.core.llm import get_llm
from src.core.schemas import Claim, SourceChunk, VerificationResult

VERIFICATION_PROMPT = """You are a citation verification agent. Given a claim and source documents, determine if the claim is supported by the sources.

Claim: {claim_text}

Source documents:
{sources}

Analyze whether the claim is:
- "supported": The source documents contain evidence that directly confirms this claim
- "contradicted": The source documents contain evidence that directly contradicts this claim
- "unsupported": The source documents do not contain relevant information about this claim

If contradicted, explain what the source actually says vs. what the claim states.

Return ONLY valid JSON:
{{"verdict": "supported|contradicted|unsupported", "confidence": 0.0-1.0, "explanation": "brief explanation"}}"""


class CitationVerifier:
    def __init__(self, indexer: CorpusIndexer):
        self.indexer = indexer
        self.llm = get_llm()

    def verify_claim(self, claim: Claim) -> VerificationResult:
        search_results = self.indexer.search(claim.text, top_k=3)

        source_chunks = [
            SourceChunk(
                content=chunk.content,
                source_doc=chunk.source_doc,
                similarity_score=score,
            )
            for chunk, score in search_results
        ]

        sources_text = "\n\n".join(
            f"[Source: {sc.source_doc} | Relevance: {sc.similarity_score:.2f}]\n{sc.content}"
            for sc in source_chunks
        )

        response = self.llm.invoke(
            VERIFICATION_PROMPT.format(claim_text=claim.text, sources=sources_text)
        )

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        parsed = json.loads(raw)

        return VerificationResult(
            claim=claim,
            verdict=parsed["verdict"],
            confidence=parsed["confidence"],
            source_chunks=source_chunks,
            explanation=parsed.get("explanation", ""),
        )

    def verify_all(self, claims: list[Claim]) -> list[VerificationResult]:
        return [self.verify_claim(c) for c in claims]
