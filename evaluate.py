"""Evaluation harness — measures hallucination detection accuracy against labeled samples."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.core.pipeline import build_pipeline
from src.core.schemas import AgentOutput, ConfidenceReport, EscalationLevel


def load_samples() -> list[dict]:
    return json.loads(Path("data/samples/sample_outputs.json").read_text())


def expected_escalation(sample: dict) -> EscalationLevel:
    verdict = sample["metadata"]["expected_verdict"]
    hal_count = sample["metadata"]["hallucination_count"]
    if verdict == "fully_supported":
        return EscalationLevel.PASS
    if verdict in ("mostly_fabricated", "fully_fabricated") or hal_count >= 4:
        return EscalationLevel.BLOCK
    return EscalationLevel.FLAG


def run_evaluation():
    samples = load_samples()
    graph, indexer = build_pipeline()
    compiled = graph.compile()

    results = []
    total_time = 0

    print(f"{'='*80}")
    print(f"AGENT OUTPUT VALIDATOR — EVALUATION RUN")
    print(f"{'='*80}")
    print(f"Samples: {len(samples)}")
    print(f"Corpus: {len(list(Path('data/corpus').glob('*.txt')))} documents")
    print(f"{'='*80}\n")

    for i, sample in enumerate(samples):
        agent_output = AgentOutput(
            id=sample["id"],
            agent_name=sample["agent_name"],
            content=sample["content"],
        )

        start = time.time()
        result = compiled.invoke({
            "agent_output": agent_output,
            "claims": [],
            "verifications": [],
            "report": None,
        })
        elapsed = time.time() - start
        total_time += elapsed

        report: ConfidenceReport = result["report"]
        expected = expected_escalation(sample)
        escalation_correct = report.escalation == expected

        expected_hals = sample["metadata"]["hallucination_count"]
        detected_hals = report.contradicted_count + report.unsupported_count

        results.append({
            "id": sample["id"],
            "expected_verdict": sample["metadata"]["expected_verdict"],
            "expected_escalation": expected.value,
            "actual_escalation": report.escalation.value,
            "escalation_correct": escalation_correct,
            "expected_hallucinations": expected_hals,
            "detected_issues": detected_hals,
            "confidence": report.overall_confidence,
            "total_claims": report.total_claims,
            "time_seconds": round(elapsed, 2),
        })

        status = "✓" if escalation_correct else "✗"
        print(f"  [{i+1:2d}/{len(samples):2d}] {status} {sample['id']:35s} "
              f"expected={expected.value:5s} actual={report.escalation.value:5s} "
              f"conf={report.overall_confidence:.0%} "
              f"hals={detected_hals}/{expected_hals} "
              f"({elapsed:.1f}s)")

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    correct = sum(1 for r in results if r["escalation_correct"])
    total = len(results)
    accuracy = correct / total

    clean_samples = [r for r in results if r["expected_hallucinations"] == 0]
    dirty_samples = [r for r in results if r["expected_hallucinations"] > 0]

    true_positives = sum(1 for r in dirty_samples if r["actual_escalation"] != "pass")
    false_negatives = sum(1 for r in dirty_samples if r["actual_escalation"] == "pass")
    true_negatives = sum(1 for r in clean_samples if r["actual_escalation"] == "pass")
    false_positives = sum(1 for r in clean_samples if r["actual_escalation"] != "pass")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Escalation accuracy:  {correct}/{total} ({accuracy:.0%})")
    print(f"  Precision:            {precision:.0%} (flagged/blocked outputs that truly had errors)")
    print(f"  Recall:               {recall:.0%} (hallucinated outputs correctly caught)")
    print(f"  F1 Score:             {f1:.2f}")
    print(f"\n  True positives:       {true_positives} (correctly flagged bad outputs)")
    print(f"  True negatives:       {true_negatives} (correctly passed clean outputs)")
    print(f"  False positives:      {false_positives} (clean outputs incorrectly flagged)")
    print(f"  False negatives:      {false_negatives} (bad outputs incorrectly passed)")

    total_expected = sum(r["expected_hallucinations"] for r in results)
    total_detected = sum(r["detected_issues"] for r in results)
    print(f"\n  Total hallucinations expected:  {total_expected}")
    print(f"  Total issues detected:         {total_detected}")

    print(f"\n  Average confidence (clean):    {sum(r['confidence'] for r in clean_samples)/len(clean_samples):.0%}")
    print(f"  Average confidence (dirty):    {sum(r['confidence'] for r in dirty_samples)/len(dirty_samples):.0%}")
    print(f"  Total evaluation time:         {total_time:.1f}s")
    print(f"  Average per sample:            {total_time/total:.1f}s")

    output_path = Path("data/evaluation_results.json")
    output_path.write_text(json.dumps({
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {
            "escalation_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_expected_hallucinations": total_expected,
            "total_detected_issues": total_detected,
            "avg_time_per_sample": round(total_time / total, 2),
        },
        "per_sample": results,
    }, indent=2))
    print(f"\n  Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_evaluation()
