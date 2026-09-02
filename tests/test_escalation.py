from src.agents.escalation_router import build_report
from src.core.schemas import (
    Claim,
    EscalationLevel,
    FailureCategory,
    VerificationResult,
)


def _make_verification(verdict: str, confidence: float, failure: FailureCategory | None = None):
    return VerificationResult(
        claim=Claim(id=0, text="test claim", claim_type="number", source_sentence="test"),
        verdict=verdict,
        confidence=confidence,
        source_chunks=[],
        failure_category=failure,
    )


def test_all_supported_passes():
    verifications = [
        _make_verification("supported", 0.95),
        _make_verification("supported", 0.90),
        _make_verification("supported", 0.85),
    ]
    report = build_report("test-pass", verifications)
    assert report.escalation == EscalationLevel.PASS
    assert report.supported_count == 3
    assert report.contradicted_count == 0
    assert report.overall_confidence >= 0.8


def test_mixed_results_flags():
    verifications = [
        _make_verification("supported", 0.9),
        _make_verification("supported", 0.85),
        _make_verification("unsupported", 0.5, FailureCategory.UNSUPPORTED_INFERENCE),
    ]
    report = build_report("test-flag", verifications)
    assert report.escalation == EscalationLevel.FLAG
    assert report.unsupported_count == 1


def test_majority_contradicted_blocks():
    verifications = [
        _make_verification("contradicted", 0.15, FailureCategory.INCORRECT_NUMBER),
        _make_verification("contradicted", 0.1, FailureCategory.FABRICATED_FACT),
        _make_verification("supported", 0.9),
    ]
    report = build_report("test-block", verifications)
    assert report.escalation == EscalationLevel.BLOCK
    assert report.contradicted_count == 2


def test_low_confidence_blocks():
    verifications = [
        _make_verification("unsupported", 0.3, FailureCategory.FABRICATED_FACT),
        _make_verification("contradicted", 0.2, FailureCategory.INCORRECT_NUMBER),
        _make_verification("unsupported", 0.4, FailureCategory.UNSUPPORTED_INFERENCE),
    ]
    report = build_report("test-low", verifications)
    assert report.escalation == EscalationLevel.BLOCK
    assert report.overall_confidence < 0.5


def test_empty_claims_passes():
    report = build_report("test-empty", [])
    assert report.escalation == EscalationLevel.PASS
    assert report.total_claims == 0
    assert report.overall_confidence == 1.0


def test_failure_breakdown_counts():
    verifications = [
        _make_verification("contradicted", 0.2, FailureCategory.INCORRECT_NUMBER),
        _make_verification("contradicted", 0.15, FailureCategory.INCORRECT_NUMBER),
        _make_verification("unsupported", 0.4, FailureCategory.FABRICATED_FACT),
        _make_verification("supported", 0.9),
    ]
    report = build_report("test-breakdown", verifications)
    assert report.failure_breakdown["incorrect_number"] == 2
    assert report.failure_breakdown["fabricated_fact"] == 1
    assert "misattribution" not in report.failure_breakdown


def test_summary_contains_key_info():
    verifications = [
        _make_verification("supported", 0.9),
        _make_verification("contradicted", 0.2, FailureCategory.INCORRECT_NUMBER),
    ]
    report = build_report("test-summary", verifications)
    assert "2 claims" in report.summary
    assert "1 supported" in report.summary
    assert "1 contradicted" in report.summary
