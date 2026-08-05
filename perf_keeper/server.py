"""REST API server for the perf-keeper diagnosis agent."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from perf_keeper.agent import create_agent
from perf_keeper.commit_triage import pipeline
from perf_keeper.commit_triage.jira_client import JiraClient
from perf_keeper.commit_triage.llm import (
    GEMINI_FLASH,
    GEMINI_PRO,
    LLMClient,
    create_client,
)
from perf_keeper.commit_triage.models import (
    CommitAnalysisRequest,
    CommitAnalysisResponse,
)

logger = logging.getLogger(__name__)

_agent: Any = None
_flash_client: LLMClient | None = None
_frontier_client: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = create_agent()
    yield


app = FastAPI(title="Perf Keeper", lifespan=lifespan)


class AgentData(BaseModel):
    job_url: HttpUrl


class AgentResponse(BaseModel):
    passed: bool
    analysis: str
    analysis_duration_seconds: int


@app.post("/analyze", response_model=AgentResponse)
async def analyze(req: AgentData):
    logger.info("Received analysis request for %s", req.job_url)
    start_time = time.time()
    state = await _agent.ainvoke({"job_url": str(req.job_url)})
    passed = state["passed"]
    elapsed_time = time.time() - start_time
    if passed:
        return AgentResponse(
            passed=True,
            analysis="Job passed. No diagnosis required.",
            analysis_duration_seconds=int(elapsed_time),
        )
    final = (state.get("final_report") or "").strip()
    return AgentResponse(
        passed=False, analysis=final, analysis_duration_seconds=int(elapsed_time)
    )


def _llm_clients() -> tuple[LLMClient, LLMClient]:
    """Return (flash, frontier), initializing once on first call."""
    global _flash_client, _frontier_client
    if _flash_client is None or _frontier_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
        _flash_client = create_client(GEMINI_FLASH, api_key=api_key)
        _frontier_client = create_client(GEMINI_PRO, api_key=api_key)
    return _flash_client, _frontier_client


def _jira_client() -> JiraClient:
    """Instantiate a JiraClient from environment variables."""
    server = os.environ.get("JIRA_SERVER", "")
    if not server:
        raise HTTPException(status_code=500, detail="JIRA_SERVER not configured")
    return JiraClient(
        server=server,
        email=os.environ.get("JIRA_EMAIL"),
        api_token=os.environ.get("JIRA_API_TOKEN"),
    )


@app.post("/analyze-commits", response_model=CommitAnalysisResponse)
async def analyze_commits(req: CommitAnalysisRequest):
    logger.info("Phase 2 request: %s", req.jira_key)
    start = time.time()

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")

    flash, frontier = _llm_clients()
    jira = _jira_client()

    try:
        await pipeline.run_for_jira(req.jira_key, jira, github_token, flash, frontier)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

    return CommitAnalysisResponse(
        jira_key=req.jira_key,
        status="ok",
        duration_seconds=int(time.time() - start),
    )
