from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.indexer import CorpusIndexer
from src.core.pipeline import build_pipeline
from src.core.schemas import AgentOutput, ConfidenceReport

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph, indexer = build_pipeline()
    _app_state["graph"] = graph
    _app_state["indexer"] = indexer
    _app_state["compiled"] = graph.compile()
    yield
    _app_state.clear()


app = FastAPI(
    title="Agent Output Validator",
    description="Validates AI agent outputs for hallucinations using a multi-agent swarm",
    version="0.1.0",
    lifespan=lifespan,
)


class ValidateRequest(BaseModel):
    id: str
    agent_name: str
    content: str
    metadata: dict = {}


class BatchRequest(BaseModel):
    outputs: list[ValidateRequest]


class BatchResponse(BaseModel):
    reports: list[ConfidenceReport]
    total: int
    passed: int
    flagged: int
    blocked: int
    avg_confidence: float


@app.post("/validate", response_model=ConfidenceReport)
def validate(req: ValidateRequest):
    compiled = _app_state.get("compiled")
    if not compiled:
        raise HTTPException(500, "Pipeline not initialized")

    agent_output = AgentOutput(
        id=req.id,
        agent_name=req.agent_name,
        content=req.content,
        metadata=req.metadata,
    )

    result = compiled.invoke({
        "agent_output": agent_output,
        "claims": [],
        "verifications": [],
        "report": None,
    })

    return result["report"]


@app.post("/validate/batch", response_model=BatchResponse)
def validate_batch(req: BatchRequest):
    compiled = _app_state.get("compiled")
    if not compiled:
        raise HTTPException(500, "Pipeline not initialized")

    reports = []
    for item in req.outputs:
        agent_output = AgentOutput(
            id=item.id,
            agent_name=item.agent_name,
            content=item.content,
            metadata=item.metadata,
        )
        result = compiled.invoke({
            "agent_output": agent_output,
            "claims": [],
            "verifications": [],
            "report": None,
        })
        reports.append(result["report"])

    passed = sum(1 for r in reports if r.escalation.value == "pass")
    flagged = sum(1 for r in reports if r.escalation.value == "flag")
    blocked = sum(1 for r in reports if r.escalation.value == "block")
    avg_conf = sum(r.overall_confidence for r in reports) / len(reports) if reports else 0

    return BatchResponse(
        reports=reports,
        total=len(reports),
        passed=passed,
        flagged=flagged,
        blocked=blocked,
        avg_confidence=round(avg_conf, 4),
    )


@app.get("/samples")
def list_samples():
    samples_path = Path("data/samples/sample_outputs.json")
    return json.loads(samples_path.read_text())


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": "indexer" in _app_state}
