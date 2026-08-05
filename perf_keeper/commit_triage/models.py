"""Phase 2 commit triage data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from perf_keeper.models import ConfidenceLevel


# enums


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


class ReviewStatus(str, Enum):
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    DISMISSED = "DISMISSED"
    PENDING = "PENDING"


# PR review data


class PRComment(BaseModel):
    body: str = ""
    created_at: datetime | None = None


class ReviewSummary(BaseModel):
    state: ReviewStatus | None = None
    body: str = ""
    submitted_at: datetime | None = None


class ReviewComment(BaseModel):
    body: str = ""
    line: int | None = None
    created_at: datetime | None = None


# commit and file models


class FileModel(BaseModel):
    """File changed in a commit. key = f"{commit_key}:{path}"."""

    file_key: str = Field(min_length=1)
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
    """Commit within a PR. key = f"{pr_key}:{sha}"."""

    commit_key: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    message: str = Field(min_length=1)
    pr_key: str = Field(min_length=1)
    committed_at: datetime | None = None
    file_keys: list[str] = Field(default_factory=list)
    total_additions: int = Field(default=0, ge=0)
    total_deletions: int = Field(default=0, ge=0)
    total_files_changed: int = Field(default=0, ge=0)
    passed_heuristics: bool = True
    triage_score: int = Field(default=0, ge=0, le=100)
    needs_deep_analysis: bool = False
    confidence: ConfidenceLevel | None = None


class PRModel(BaseModel):
    """Pull request. key = f"{owner}/{repo}#{number}"."""

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


# regression context


class AffectedMetric(BaseModel):
    name: str = Field(min_length=1)
    change: float
    value: float | None = None


class MetricContext(BaseModel):
    """Optional domain knowledge for heuristic PR filtering."""

    name: str = Field(min_length=1)
    related_repos: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    known_causes: list[str] = Field(default_factory=list)


class RegressionContext(BaseModel):
    """Complete context for a Phase 2 triage run. Uses flat dicts for O(1) lookups."""

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


# LLM triage output


class FlashDecision(BaseModel):
    commit_key: str
    worth_investigating: bool


class FlashResponse(BaseModel):
    decisions: list[FlashDecision]


class CommitRanking(BaseModel):
    commit_key: str
    triage_score: int
    confidence: ConfidenceLevel
    reasoning: str


class FrontierResponse(BaseModel):
    rankings: list[CommitRanking]


# pipeline I/O


class CommitAnalysisRequest(BaseModel):
    """Request body for POST /analyze-commits."""

    jira_key: str = Field(min_length=1)


class CommitAnalysisResponse(BaseModel):
    """Response from POST /analyze-commits."""

    jira_key: str = Field(min_length=1)
    status: str = "ok"
    duration_seconds: int = 0
