from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """6-category failure taxonomy (based on GreenPepper LLM eval framework)."""
    FABRICATED_FACT = "fabricated_fact"
    INCORRECT_NUMBER = "incorrect_number"
    MISATTRIBUTION = "misattribution"
    TEMPORAL_ERROR = "temporal_error"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    CONTRADICTED_BY_SOURCE = "contradicted_by_source"


class EscalationLevel(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


class Claim(BaseModel):
    """A single factual claim extracted from an agent's output."""
    id: int
    text: str
    claim_type: str = Field(description="number, date, ticker, percentage, entity, or general")
    source_sentence: str = Field(description="The full sentence this claim was extracted from")


class SourceChunk(BaseModel):
    """A chunk retrieved from the source corpus for verification."""
    content: str
    source_doc: str
    similarity_score: float


class VerificationResult(BaseModel):
    """Result of verifying a single claim against source documents."""
    claim: Claim
    verdict: str = Field(description="supported, contradicted, or unsupported")
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunks: list[SourceChunk] = Field(default_factory=list)
    failure_category: FailureCategory | None = None
    explanation: str = ""


class ConfidenceReport(BaseModel):
    """Final output of the validation pipeline."""
    agent_output_id: str
    overall_confidence: float = Field(ge=0.0, le=1.0)
    escalation: EscalationLevel
    total_claims: int
    supported_count: int
    contradicted_count: int
    unsupported_count: int
    failure_breakdown: dict[str, int] = Field(default_factory=dict)
    verifications: list[VerificationResult]
    summary: str


class AgentOutput(BaseModel):
    """Input to the validation pipeline — an output produced by any DAF agent."""
    id: str
    agent_name: str
    content: str
    metadata: dict = Field(default_factory=dict)


class PipelineState(BaseModel):
    """LangGraph state that flows through the validation pipeline."""
    agent_output: AgentOutput
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    report: ConfidenceReport | None = None
