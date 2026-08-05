from __future__ import annotations

from enum import Enum

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


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RegressionType(str, Enum):
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    INFRASTRUCTURE = "infrastructure"


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
