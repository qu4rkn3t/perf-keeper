"""Shared fixtures for commit_triage tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from perf_keeper.commit_triage.llm import GEMINI_FLASH, LLMClient
from perf_keeper.commit_triage.models import (
    AffectedMetric,
    CommitModel,
    FileModel,
    FileStatus,
    MetricContext,
    PRComment,
    PRModel,
    RegressionContext,
    ReviewStatus,
    ReviewSummary,
)

T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2025, 1, 30, 0, 0, 0, tzinfo=timezone.utc)

PR_KEY = "openshift/kubernetes#1234"
SHA = "abc12345def56789"
COMMIT_KEY = f"{PR_KEY}:{SHA}"
FILE_PATH = "pkg/kubelet/pod_manager.go"
FILE_KEY = f"{COMMIT_KEY}:{FILE_PATH}"

SIMPLE_DIFF = (
    "@@ -10,3 +10,6 @@ func (m *Manager) syncPod(pod *v1.Pod) error {\n"
    "-\treturn m.runtime.SyncPod(pod)\n"
    "+\tif pod == nil {\n"
    '+\t\treturn fmt.Errorf("nil pod")\n'
    "+\t}\n"
    "+\treturn m.runtime.SyncPod(pod)\n"
)


class MockLLMClient(LLMClient):
    """Deterministic LLM client for tests — no real API calls."""

    def __init__(self, context_window: int = 10_000) -> None:
        self.model = GEMINI_FLASH
        self._api_key = "test"
        self._temperature = 0.0
        self.context_window = context_window

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def complete(self, prompt: str) -> str:
        return ""

    async def complete_structured(self, prompt: str, schema):
        return schema.model_construct()

    async def embed(self, text: str) -> list[float]:
        h = abs(hash(text)) % 1000
        v = h / 1000.0
        return [v, 1.0 - v]


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def primary_metric() -> AffectedMetric:
    return AffectedMetric(name="podReadyLatency", change=-15.0, value=250.0)


@pytest.fixture
def metric_context() -> MetricContext:
    return MetricContext(
        name="podReadyLatency",
        related_repos=["openshift/kubernetes"],
        description="time for pods to become ready",
        known_causes=["kubelet performance", "scheduler changes"],
    )


@pytest.fixture
def regression_context(
    tmp_path: Path, primary_metric: AffectedMetric
) -> RegressionContext:
    diff_p = tmp_path / "diffs" / SHA[:8] / "abcd1234.diff"
    diff_p.parent.mkdir(parents=True, exist_ok=True)
    diff_p.write_text(SIMPLE_DIFF, encoding="utf-8")

    file_model = FileModel(
        file_key=FILE_KEY,
        commit_key=COMMIT_KEY,
        path=FILE_PATH,
        diff_path=diff_p,
        additions=4,
        deletions=1,
        status=FileStatus.MODIFIED,
    )
    commit = CommitModel(
        commit_key=COMMIT_KEY,
        sha=SHA,
        message="Optimize pod sync in kubelet manager",
        pr_key=PR_KEY,
        committed_at=T1,
        file_keys=[FILE_KEY],
        total_additions=4,
        total_deletions=1,
        total_files_changed=1,
        passed_heuristics=True,
    )
    pr = PRModel(
        pr_key=PR_KEY,
        owner="openshift",
        repo="kubernetes",
        number=1234,
        title="Optimize pod scheduling and kubelet sync performance",
        description="Reduces unnecessary allocations in pod sync path",
        commit_keys=[COMMIT_KEY],
        labels=["performance", "kubelet"],
        comments=[PRComment(body="LGTM", created_at=T0)],
        review_summaries=[
            ReviewSummary(state=ReviewStatus.APPROVED, body="Approved", submitted_at=T1)
        ],
        total_additions=4,
        total_deletions=1,
        total_files_changed=1,
    )
    report_path = tmp_path / "report.md"
    report_path.write_text("# Phase 1 Report\n")

    return RegressionContext(
        analysis_id="test-001",
        workspace_dir=tmp_path,
        jira_key="PERFSCALE-1234",
        jira_title="Pod ready latency regression in 4.18",
        jira_labels=["regression", "4.18"],
        test="node-density",
        primary_metric=primary_metric,
        all_affected_metrics=[
            AffectedMetric(name="apiserverCPU", change=10.0, value=45.0)
        ],
        good_version="4.17.0-0.nightly-2025-01-01",
        bad_version="4.18.0-0.nightly-2025-01-15",
        perf_keeper_report_path=report_path,
        prs_by_key={PR_KEY: pr},
        commits_by_key={COMMIT_KEY: commit},
        files_by_key={FILE_KEY: file_model},
    )
