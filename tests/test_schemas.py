from src.core.schemas import (
    AgentOutput,
    Claim,
    ConfidenceReport,
    EscalationLevel,
    FailureCategory,
    PipelineState,
    SourceChunk,
    VerificationResult,
)


def test_claim_creation():
    claim = Claim(
        id=0,
        text="Revenue was $215.938 billion",
        claim_type="number",
        source_sentence="NVIDIA reported revenue of $215.938 billion.",
    )
    assert claim.id == 0
    assert claim.claim_type == "number"


def test_verification_result_supported():
    claim = Claim(id=0, text="test", claim_type="general", source_sentence="test sentence")
    result = VerificationResult(
        claim=claim,
        verdict="supported",
        confidence=0.95,
        source_chunks=[],
    )
    assert result.verdict == "supported"
    assert result.failure_category is None


def test_verification_result_contradicted():
    claim = Claim(id=1, text="AUM was $4.1B", claim_type="number", source_sentence="AUM was $4.1B")
    result = VerificationResult(
        claim=claim,
        verdict="contradicted",
        confidence=0.2,
        failure_category=FailureCategory.INCORRECT_NUMBER,
        source_chunks=[],
        explanation="Source says $3.2B",
    )
    assert result.verdict == "contradicted"
    assert result.failure_category == FailureCategory.INCORRECT_NUMBER


def test_agent_output():
    output = AgentOutput(
        id="test-001",
        agent_name="EarningsAgent",
        content="NVIDIA reported $215.938 billion in revenue.",
    )
    assert output.id == "test-001"
    assert output.metadata == {}


def test_pipeline_state_defaults():
    output = AgentOutput(id="x", agent_name="test", content="test content")
    state = PipelineState(agent_output=output)
    assert state.claims == []
    assert state.verifications == []
    assert state.report is None


def test_failure_category_values():
    assert len(FailureCategory) == 6
    assert FailureCategory.FABRICATED_FACT.value == "fabricated_fact"
    assert FailureCategory.INCORRECT_NUMBER.value == "incorrect_number"
    assert FailureCategory.MISATTRIBUTION.value == "misattribution"
    assert FailureCategory.TEMPORAL_ERROR.value == "temporal_error"
    assert FailureCategory.UNSUPPORTED_INFERENCE.value == "unsupported_inference"
    assert FailureCategory.CONTRADICTED_BY_SOURCE.value == "contradicted_by_source"
