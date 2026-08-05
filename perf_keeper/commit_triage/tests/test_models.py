"""Tests for commit_triage data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from perf_keeper.commit_triage.models import (
    AffectedMetric,
    CommitAnalysisRequest,
    CommitAnalysisResponse,
    CommitModel,
    CommitRanking,
    ConfidenceLevel,
    DiscardReason,
    FileModel,
    FileStatus,
    FlashDecision,
    FlashResponse,
    FrontierResponse,
    MetricContext,
    PRComment,
    PRModel,
    RegressionContext,
    ReviewComment,
    ReviewStatus,
    ReviewSummary,
)
from perf_keeper.models import ConfidenceLevel as RootConfidenceLevel


# enums


class TestEnums:
    def test_file_status_values(self):
        assert FileStatus.ADDED == "added"
        assert FileStatus.MODIFIED == "modified"
        assert FileStatus.REMOVED == "removed"
        assert FileStatus.RENAMED == "renamed"

    def test_discard_reason_values(self):
        assert DiscardReason.BINARY == "binary"
        assert DiscardReason.TEST_FILE == "test_file"
        assert DiscardReason.DOCS == "docs"
        assert DiscardReason.CONFIG == "config"

    def test_confidence_level_values(self):
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.HIGH == "high"

    def test_confidence_level_shared_with_root(self):
        assert ConfidenceLevel is RootConfidenceLevel

    def test_review_status_values(self):
        assert ReviewStatus.APPROVED == "APPROVED"
        assert ReviewStatus.CHANGES_REQUESTED == "CHANGES_REQUESTED"
        assert ReviewStatus.COMMENTED == "COMMENTED"


# PR review data


class TestPRComment:
    def test_defaults(self):
        c = PRComment()
        assert c.body == ""
        assert c.created_at is None

    def test_with_values(self):
        from datetime import datetime, timezone

        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        c = PRComment(body="LGTM", created_at=dt)
        assert c.body == "LGTM"
        assert c.created_at == dt


class TestReviewSummary:
    def test_defaults(self):
        r = ReviewSummary()
        assert r.state is None
        assert r.body == ""

    def test_with_state(self):
        r = ReviewSummary(state=ReviewStatus.APPROVED, body="Ship it")
        assert r.state == ReviewStatus.APPROVED


class TestReviewComment:
    def test_defaults(self):
        r = ReviewComment()
        assert r.body == ""
        assert r.line is None

    def test_with_line(self):
        r = ReviewComment(body="nit", line=42)
        assert r.line == 42


# commit and file models


class TestFileModel:
    def test_construction(self, tmp_path):
        diff_p = tmp_path / "test.diff"
        m = FileModel(
            file_key="pr:sha:path.py",
            commit_key="pr:sha",
            path="path.py",
            diff_path=diff_p,
        )
        assert m.additions == 0
        assert m.deletions == 0
        assert m.discarded is False
        assert m.discard_reason is None
        assert m.status is None
        assert m.review_comments == []

    def test_empty_file_key_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            FileModel(
                file_key="",
                commit_key="pr:sha",
                path="p.py",
                diff_path=tmp_path / "x.diff",
            )

    def test_negative_additions_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            FileModel(
                file_key="a:b:c",
                commit_key="a:b",
                path="c",
                diff_path=tmp_path / "x.diff",
                additions=-1,
            )

    def test_status_enum(self, tmp_path):
        m = FileModel(
            file_key="a:b:c",
            commit_key="a:b",
            path="c",
            diff_path=tmp_path / "x.diff",
            status=FileStatus.ADDED,
        )
        assert m.status == FileStatus.ADDED

    def test_discard_fields(self, tmp_path):
        m = FileModel(
            file_key="a:b:c",
            commit_key="a:b",
            path="c",
            diff_path=tmp_path / "x.diff",
            discarded=True,
            discard_reason=DiscardReason.BINARY,
        )
        assert m.discarded is True
        assert m.discard_reason == DiscardReason.BINARY


class TestCommitModel:
    def test_defaults(self):
        c = CommitModel(commit_key="pr:sha", sha="sha", message="fix", pr_key="pr")
        assert c.passed_heuristics is True
        assert c.triage_score == 0
        assert c.needs_deep_analysis is False
        assert c.confidence is None
        assert c.file_keys == []

    def test_triage_score_upper_bound(self):
        with pytest.raises(ValidationError):
            CommitModel(
                commit_key="p:s", sha="s", message="m", pr_key="p", triage_score=101
            )

    def test_triage_score_lower_bound(self):
        with pytest.raises(ValidationError):
            CommitModel(
                commit_key="p:s", sha="s", message="m", pr_key="p", triage_score=-1
            )

    def test_triage_score_valid_range(self):
        for score in (0, 50, 100):
            c = CommitModel(
                commit_key="p:s", sha="s", message="m", pr_key="p", triage_score=score
            )
            assert c.triage_score == score

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            CommitModel(commit_key="p:s", sha="s", message="", pr_key="p")

    def test_confidence_enum(self):
        c = CommitModel(
            commit_key="p:s",
            sha="s",
            message="m",
            pr_key="p",
            confidence=ConfidenceLevel.HIGH,
        )
        assert c.confidence == ConfidenceLevel.HIGH


class TestPRModel:
    def test_defaults(self):
        pr = PRModel(pr_key="o/r#1", owner="o", repo="r", number=1, title="Title")
        assert pr.description == ""
        assert pr.labels == []
        assert pr.comments == []
        assert pr.review_summaries == []

    def test_number_zero_raises(self):
        with pytest.raises(ValidationError):
            PRModel(pr_key="o/r#0", owner="o", repo="r", number=0, title="T")

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError):
            PRModel(pr_key="o/r#1", owner="o", repo="r", number=1, title="")

    def test_with_full_data(self):
        pr = PRModel(
            pr_key="o/r#5",
            owner="o",
            repo="r",
            number=5,
            title="Perf fix",
            labels=["performance"],
            total_additions=100,
            total_deletions=50,
        )
        assert pr.total_additions == 100
        assert pr.total_files_changed == 0


# regression context


class TestAffectedMetric:
    def test_defaults(self):
        m = AffectedMetric(name="cpu", change=-10.0)
        assert m.value is None

    def test_with_value(self):
        m = AffectedMetric(name="cpu", change=-10.0, value=45.5)
        assert m.value == 45.5

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            AffectedMetric(name="", change=0.0)


class TestMetricContext:
    def test_construction(self):
        ctx = MetricContext(
            name="ovnCPU",
            description="OVN controller CPU usage",
            related_repos=["openshift/ovn-kubernetes"],
            known_causes=["flow table growth"],
        )
        assert ctx.name == "ovnCPU"
        assert len(ctx.related_repos) == 1

    def test_empty_description_raises(self):
        with pytest.raises(ValidationError):
            MetricContext(name="x", description="", known_causes=[])


class TestRegressionContext:
    def test_construction(self, tmp_path, primary_metric):
        ctx = RegressionContext(
            analysis_id="abc",
            workspace_dir=tmp_path,
            jira_key="PERF-1",
            test="node-density",
            primary_metric=primary_metric,
            good_version="4.17",
            bad_version="4.18",
            perf_keeper_report_path=tmp_path / "r.md",
        )
        assert ctx.prs_by_key == {}
        assert ctx.commits_by_key == {}
        assert ctx.files_by_key == {}
        assert ctx.metric_context is None
        assert ctx.jira_title == ""
        assert ctx.jira_labels == []

    def test_empty_jira_key_raises(self, tmp_path, primary_metric):
        with pytest.raises(ValidationError):
            RegressionContext(
                analysis_id="abc",
                workspace_dir=tmp_path,
                jira_key="",
                test="t",
                primary_metric=primary_metric,
                good_version="4.17",
                bad_version="4.18",
                perf_keeper_report_path=tmp_path / "r.md",
            )


# LLM triage output


class TestFlashDecision:
    def test_construction(self):
        d = FlashDecision(
            commit_key="pr:sha", worth_investigating=True, reason="hot path change"
        )
        assert d.commit_key == "pr:sha"
        assert d.worth_investigating is True

    def test_not_worth_investigating(self):
        d = FlashDecision(
            commit_key="pr:sha", worth_investigating=False, reason="docs only"
        )
        assert d.worth_investigating is False


class TestFlashResponse:
    def test_empty_decisions(self):
        r = FlashResponse(decisions=[])
        assert r.decisions == []

    def test_multiple_decisions(self):
        r = FlashResponse(
            decisions=[
                FlashDecision(commit_key="a", worth_investigating=True),
                FlashDecision(commit_key="b", worth_investigating=False),
            ]
        )
        assert len(r.decisions) == 2


class TestCommitRanking:
    def test_construction(self):
        r = CommitRanking(
            commit_key="pr:sha",
            triage_score=85,
            confidence=ConfidenceLevel.HIGH,
            reasoning="logic change in hot path",
        )
        assert r.triage_score == 85
        assert r.confidence == ConfidenceLevel.HIGH

    def test_all_confidence_levels(self):
        for level in ConfidenceLevel:
            r = CommitRanking(
                commit_key="k", triage_score=50, confidence=level, reasoning="r"
            )
            assert r.confidence == level


class TestFrontierResponse:
    def test_empty_rankings(self):
        r = FrontierResponse(rankings=[])
        assert r.rankings == []

    def test_multiple_rankings(self):
        r = FrontierResponse(
            rankings=[
                CommitRanking(
                    commit_key="a",
                    triage_score=90,
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="r1",
                ),
                CommitRanking(
                    commit_key="b",
                    triage_score=60,
                    confidence=ConfidenceLevel.LOW,
                    reasoning="r2",
                ),
            ]
        )
        assert len(r.rankings) == 2
        assert r.rankings[0].triage_score > r.rankings[1].triage_score


# pipeline I/O


class TestCommitAnalysisRequest:
    def test_valid(self):
        r = CommitAnalysisRequest(jira_key="PERF-1")
        assert r.jira_key == "PERF-1"

    def test_empty_jira_key_raises(self):
        with pytest.raises(ValidationError):
            CommitAnalysisRequest(jira_key="")


class TestCommitAnalysisResponse:
    def test_valid(self):
        r = CommitAnalysisResponse(jira_key="PERF-1", analysis="# Report\n...")
        assert r.jira_key == "PERF-1"

    def test_empty_analysis_raises(self):
        with pytest.raises(ValidationError):
            CommitAnalysisResponse(jira_key="PERF-1", analysis="")
