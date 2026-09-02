from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.core.pipeline import build_pipeline
from src.core.schemas import AgentOutput, ConfidenceReport, EscalationLevel

st.set_page_config(
    page_title="Agent Output Validator",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    graph, indexer = build_pipeline()
    return graph.compile(), indexer


def render_escalation_badge(escalation: EscalationLevel):
    colors = {
        EscalationLevel.PASS: ("#16a34a", "#dcfce7", "PASS"),
        EscalationLevel.FLAG: ("#ea580c", "#fff7ed", "FLAG"),
        EscalationLevel.BLOCK: ("#dc2626", "#fef2f2", "BLOCK"),
    }
    fg, bg, label = colors[escalation]
    st.markdown(
        f'<span style="background:{bg};color:{fg};padding:4px 12px;border-radius:4px;'
        f'font-weight:700;font-size:14px;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence: float):
    color = "#16a34a" if confidence >= 0.8 else "#ea580c" if confidence >= 0.5 else "#dc2626"
    pct = int(confidence * 100)
    st.markdown(
        f'<div style="background:#e5e7eb;border-radius:6px;height:20px;width:100%;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct}%;border-radius:6px;'
        f'display:flex;align-items:center;padding-left:8px;">'
        f'<span style="color:white;font-size:11px;font-weight:600;">{pct}%</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_claim_row(v, i):
    icons = {"supported": "✅", "contradicted": "❌", "unsupported": "⚠️"}
    icon = icons.get(v.verdict, "❓")

    with st.container():
        cols = st.columns([0.5, 5, 1.5, 1])
        cols[0].markdown(f"**{i+1}**")
        cols[1].markdown(f"{icon} {v.claim.text}")
        cols[2].markdown(f"`{v.verdict}`")

        conf_color = "#16a34a" if v.confidence >= 0.8 else "#ea580c" if v.confidence >= 0.5 else "#dc2626"
        cols[3].markdown(
            f'<span style="color:{conf_color};font-weight:600;">{v.confidence:.0%}</span>',
            unsafe_allow_html=True,
        )

        if v.verdict != "supported":
            with st.expander("Details", expanded=False):
                if v.failure_category:
                    st.markdown(f"**Failure type:** `{v.failure_category.value}`")
                if v.explanation:
                    st.markdown(f"**Explanation:** {v.explanation}")
                if v.source_chunks:
                    st.markdown("**Source evidence:**")
                    for sc in v.source_chunks[:2]:
                        st.markdown(
                            f"> *[{sc.source_doc} — relevance: {sc.similarity_score:.2f}]*\n> "
                            f"{sc.content[:200]}..."
                        )


def render_report(report: ConfidenceReport):
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", report.total_claims)
    with col2:
        st.metric("Supported", report.supported_count)
    with col3:
        st.metric("Contradicted", report.contradicted_count)
    with col4:
        st.metric("Unsupported", report.unsupported_count)

    st.markdown("### Confidence")
    conf_col, esc_col = st.columns([3, 1])
    with conf_col:
        render_confidence_bar(report.overall_confidence)
    with esc_col:
        render_escalation_badge(report.escalation)

    if report.failure_breakdown:
        st.markdown("### Failure Breakdown")
        for cat, count in sorted(report.failure_breakdown.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{cat.replace('_', ' ').title()}**: {count}")

    st.markdown("### Claim-by-Claim Analysis")
    for i, v in enumerate(report.verifications):
        render_claim_row(v, i)

    st.markdown("### Summary")
    st.info(report.summary)


def main():
    st.title("🛡️ Agent Output Validator")
    st.caption("Multi-agent swarm for hallucination detection in AI agent outputs")

    compiled, indexer = load_pipeline()

    tab_single, tab_batch, tab_samples = st.tabs(["Single Validation", "Batch Mode", "Sample Gallery"])

    with tab_single:
        st.markdown("Paste an agent's output to validate it against the source corpus.")

        col_form, col_meta = st.columns([3, 1])
        with col_meta:
            agent_id = st.text_input("Output ID", value="custom-001")
            agent_name = st.text_input("Agent Name", value="CustomAgent")
        with col_form:
            content = st.text_area(
                "Agent Output",
                height=200,
                placeholder="Paste the agent's response here...",
            )

        if st.button("Validate", type="primary", disabled=not content):
            with st.spinner("Running validation pipeline..."):
                agent_output = AgentOutput(
                    id=agent_id, agent_name=agent_name, content=content
                )
                result = compiled.invoke({
                    "agent_output": agent_output,
                    "claims": [],
                    "verifications": [],
                    "report": None,
                })
                render_report(result["report"])

    with tab_batch:
        st.markdown("Upload a JSON file with multiple agent outputs to validate in batch.")

        uploaded = st.file_uploader("Upload JSON", type=["json"])
        if uploaded:
            data = json.loads(uploaded.read())
            if not isinstance(data, list):
                data = [data]

            st.markdown(f"**{len(data)} outputs loaded**")

            if st.button("Run Batch Validation", type="primary"):
                reports = []
                progress = st.progress(0)

                for i, item in enumerate(data):
                    agent_output = AgentOutput(
                        id=item["id"],
                        agent_name=item["agent_name"],
                        content=item["content"],
                        metadata=item.get("metadata", {}),
                    )
                    result = compiled.invoke({
                        "agent_output": agent_output,
                        "claims": [],
                        "verifications": [],
                        "report": None,
                    })
                    reports.append(result["report"])
                    progress.progress((i + 1) / len(data))

                st.divider()
                st.markdown("### Batch Results")

                m1, m2, m3, m4, m5 = st.columns(5)
                passed = sum(1 for r in reports if r.escalation == EscalationLevel.PASS)
                flagged = sum(1 for r in reports if r.escalation == EscalationLevel.FLAG)
                blocked = sum(1 for r in reports if r.escalation == EscalationLevel.BLOCK)
                avg = sum(r.overall_confidence for r in reports) / len(reports)

                m1.metric("Total", len(reports))
                m2.metric("Passed", passed)
                m3.metric("Flagged", flagged)
                m4.metric("Blocked", blocked)
                m5.metric("Avg Confidence", f"{avg:.0%}")

                all_failures: dict[str, int] = {}
                for r in reports:
                    for cat, count in r.failure_breakdown.items():
                        all_failures[cat] = all_failures.get(cat, 0) + count

                if all_failures:
                    st.markdown("### Aggregate Failure Types")
                    for cat, count in sorted(all_failures.items(), key=lambda x: -x[1]):
                        st.markdown(f"- **{cat.replace('_', ' ').title()}**: {count}")

                for report in reports:
                    with st.expander(
                        f"{report.agent_output_id} — {report.escalation.value.upper()} "
                        f"({report.overall_confidence:.0%})"
                    ):
                        render_report(report)

    with tab_samples:
        st.markdown("Pre-built test cases with known hallucination counts.")

        samples_path = Path("data/samples/sample_outputs.json")
        samples = json.loads(samples_path.read_text())

        for sample in samples:
            expected = sample.get("metadata", {}).get("expected_verdict", "unknown")
            hal_count = sample.get("metadata", {}).get("hallucination_count", "?")

            with st.expander(f"**{sample['id']}** ({sample['agent_name']}) — expected: {expected}, hallucinations: {hal_count}"):
                st.code(sample["content"], language=None)

                if st.button(f"Validate {sample['id']}", key=sample["id"]):
                    with st.spinner("Validating..."):
                        agent_output = AgentOutput(
                            id=sample["id"],
                            agent_name=sample["agent_name"],
                            content=sample["content"],
                        )
                        result = compiled.invoke({
                            "agent_output": agent_output,
                            "claims": [],
                            "verifications": [],
                            "report": None,
                        })
                        render_report(result["report"])


if __name__ == "__main__":
    main()
