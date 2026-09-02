"""CLI runner for the Agent Output Validator pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.pipeline import build_pipeline
from src.core.schemas import AgentOutput


def run_single(sample_id: str | None = None):
    samples_path = Path("data/samples/sample_outputs.json")
    samples = json.loads(samples_path.read_text())

    if sample_id:
        samples = [s for s in samples if s["id"] == sample_id]
        if not samples:
            print(f"Sample '{sample_id}' not found.")
            sys.exit(1)

    graph, indexer = build_pipeline()
    app = graph.compile()

    for sample in samples:
        agent_output = AgentOutput(
            id=sample["id"],
            agent_name=sample["agent_name"],
            content=sample["content"],
            metadata=sample.get("metadata", {}),
        )

        print(f"\n{'='*70}")
        print(f"Validating: {agent_output.id} ({agent_output.agent_name})")
        print(f"Content: {agent_output.content[:120]}...")
        print(f"{'='*70}")

        result = app.invoke({
            "agent_output": agent_output,
            "claims": [],
            "verifications": [],
            "report": None,
        })

        report = result["report"]
        print(f"\n  Overall Confidence: {report.overall_confidence:.1%}")
        print(f"  Escalation: {report.escalation.value.upper()}")
        print(f"  Claims: {report.total_claims} total | "
              f"{report.supported_count} supported | "
              f"{report.contradicted_count} contradicted | "
              f"{report.unsupported_count} unsupported")

        if report.failure_breakdown:
            print(f"  Failure types: {report.failure_breakdown}")

        for v in report.verifications:
            icon = {"supported": "✓", "contradicted": "✗", "unsupported": "?"}.get(v.verdict, "?")
            cat = f" [{v.failure_category.value}]" if v.failure_category else ""
            print(f"    {icon} [{v.confidence:.0%}] {v.claim.text[:80]}{cat}")
            if v.explanation:
                print(f"      → {v.explanation[:100]}")

        print(f"\n  Summary: {report.summary}")

        expected = sample.get("metadata", {}).get("expected_verdict", "unknown")
        expected_count = sample.get("metadata", {}).get("hallucination_count", "?")
        print(f"  Expected: {expected} (hallucinations: {expected_count})")


if __name__ == "__main__":
    sample_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_single(sample_id)
