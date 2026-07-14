from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStage(str, Enum):
    CLUSTER_INSTALL = "cluster_install"
    DAY2_OPS = "day2_ops"
    WORKLOAD = "workload"
    TEARDOWN = "teardown"


class JobStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class FileStatus(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"


class DiscardReason(str, Enum):
    BINARY = "binary"
    TEST_FILE = "test_file"
    DOCS = "docs"
    CONFIG = "config"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RegressionType(str, Enum):
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    INFRASTRUCTURE = "infrastructure"


class ReviewStatus(str, Enum):
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    DISMISSED = "DISMISSED"
    PENDING = "PENDING"


class ProwJob(BaseModel):
    """Represents a single prow job run."""

    job_name: str = Field(min_length=1)
    build_id: str = Field(min_length=1)
    url: str = ""
    status: JobStatus = JobStatus.UNKNOWN
    failed_stage: JobStage | None = None
    ocp_version: str = ""
    payload: str = ""
    duration_minutes: float = Field(default=0.0, ge=0.0)
    artifacts_url: str = ""


class PullRequestInfo(BaseModel):
    """A PR included in a payload diff."""

    repo: str = Field(min_length=1)
    number: int = Field(gt=0)
    title: str = Field(min_length=1)
    author: str = ""
    url: str = ""
    merged_at: str = ""
    files_changed: list[str] = Field(default_factory=list)
    description: str = ""


class ComponentChange(BaseModel):
    """A component that changed between payloads."""

    name: str = Field(min_length=1)
    from_image: str = ""
    to_image: str = ""
    pull_requests: list[PullRequestInfo] = Field(default_factory=list)


class PayloadDiff(BaseModel):
    """Diff between two OCP payloads from Sippy."""

    from_payload: str = Field(min_length=1)
    to_payload: str = Field(min_length=1)
    ocp_version: str = Field(min_length=1)
    pull_requests: list[PullRequestInfo] = Field(default_factory=list)
    component_changes: list[ComponentChange] = Field(default_factory=list)


class MetricSample(BaseModel):
    """A performance metric sample from kube-burner or similar."""

    metric_name: str = Field(min_length=1)
    value: float
    unit: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: str = ""


class JobMetrics(BaseModel):
    """Aggregated metrics from a prow job run."""

    job: ProwJob
    metrics: list[MetricSample] = Field(default_factory=list)
    alerts_fired: list[str] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    """Final diagnosis output from the agent."""

    summary: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    evidence: list[str] = Field(default_factory=list)
    suspect_prs: list[PullRequestInfo] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    regression_type: RegressionType | None = None
    affected_metrics: list[MetricSample] = Field(default_factory=list)


class PRComment(BaseModel):
    """A comment on a pull request."""

    body: str = ""
    created_at: datetime | None = None


class ReviewSummary(BaseModel):
    """Top-level review submitted on a pull request."""

    state: ReviewStatus | None = None
    body: str = ""
    submitted_at: datetime | None = None


class ReviewComment(BaseModel):
    """Inline review comment on a specific file in a pull request."""

    body: str = ""
    line: int | None = None
    created_at: datetime | None = None


class FileModel(BaseModel):
    """File changed in a commit. Stored by key = f"{commit_key}:{path}"."""

    file_key: str = Field(min_length=1)  # f"{commit_key}:{path}"
    commit_key: str = Field(min_length=1)
    path: str = Field(min_length=1)
    diff_path: Path
    review_comments: list[ReviewComment] = Field(default_factory=list)
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    status: FileStatus | None = None
    discarded: bool = False
    discard_reason: DiscardReason | None = None


class CommitModel(BaseModel):
    """Commit within a PR. Stored by key = f"{pr_key}:{commit_sha}"."""

    commit_key: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    message: str = Field(min_length=1)
    pr_key: str = Field(min_length=1)
    file_keys: list[str] = Field(default_factory=list)
    total_additions: int = Field(default=0, ge=0)
    total_deletions: int = Field(default=0, ge=0)
    total_files_changed: int = Field(default=0, ge=0)
    passed_heuristics: bool = True
    triage_score: int = Field(default=0, ge=0, le=100)
    needs_deep_analysis: bool = False


class PRModel(BaseModel):
    """Pull request in commit triage analysis. Stored by key = f"{owner}/{repo}#{number}"."""

    pr_key: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    number: int = Field(gt=0)
    title: str = Field(min_length=1)
    description: str = ""
    commit_keys: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    comments: list[PRComment] = Field(default_factory=list)
    review_summaries: list[ReviewSummary] = Field(default_factory=list)
    total_additions: int = Field(default=0, ge=0)
    total_deletions: int = Field(default=0, ge=0)
    total_files_changed: int = Field(default=0, ge=0)


class AffectedMetric(BaseModel):
    """A metric affected by a regression."""

    name: str = Field(min_length=1)
    change: float
    value: float | None = None


class MetricContext(BaseModel):
    """Optional domain knowledge for PR filtering in heuristics."""

    name: str = Field(min_length=1)
    related_repos: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    known_causes: list[str] = Field(default_factory=list)


class RegressionContext(BaseModel):
    """Complete context for Phase 2 commit triage. Uses flat dicts for O(1) lookups."""

    analysis_id: str = Field(min_length=1)
    workspace_dir: Path
    jira_key: str = Field(min_length=1)
    jira_title: str = ""
    jira_labels: list[str] = Field(default_factory=list)
    test: str = Field(min_length=1)
    primary_metric: AffectedMetric
    all_affected_metrics: list[AffectedMetric] = Field(default_factory=list)
    good_version: str = Field(min_length=1)
    bad_version: str = Field(min_length=1)
    perf_keeper_report_path: Path
    prs_by_key: dict[str, PRModel] = Field(default_factory=dict)
    commits_by_key: dict[str, CommitModel] = Field(default_factory=dict)
    files_by_key: dict[str, FileModel] = Field(default_factory=dict)
    metric_context: MetricContext | None = None


class CommitAnalysisRequest(BaseModel):
    """Request for /analyze-commits."""

    jira_key: str = Field(min_length=1)


class CommitAnalysisResponse(BaseModel):
    """Response from /analyze-commits. Returns combined markdown (perf-keeper + commits)."""

    jira_key: str = Field(min_length=1)
    analysis: str = Field(min_length=1)
