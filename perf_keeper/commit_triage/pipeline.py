"""Phase 2 commit triage pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perf_keeper.commit_triage.jira_client import JiraClient

from perf_keeper.commit_triage.config import (
    MAX_FRONTIER_RANKINGS,
    MIN_PASSING_COMMITS,
    NEEDS_DEEP_ANALYSIS_TOP_N,
    S6_WEIGHT,
    SCORE_THRESHOLD,
    SIGNAL_WEIGHTS,
)
from perf_keeper.commit_triage.heuristics import (
    s1_temporal_proximity,
    s2_modification_intensity,
    s3_nesting_depth_shift,
    s4_control_flow_delta,
    s5_change_concentration,
    s6_component_proximity,
    should_discard,
)
from perf_keeper.commit_triage.llm import LLMClient
from perf_keeper.commit_triage.models import (
    CommitModel,
    FlashResponse,
    FrontierResponse,
    RegressionContext,
)
from perf_keeper.commit_triage.prompts import (
    build_flash_prompts,
    build_frontier_prompts,
    load_all_diffs,
)
from perf_keeper.commit_triage.report import save_report

logger = logging.getLogger(__name__)


def _combine_signals(
    s1: float, s2: float, s3: float, s4: float, s5: float, s6: float | None
) -> int:
    """Weighted average of S1-S5 (always) plus S6 (only when MetricContext is present)."""
    signals = [s1, s2, s3, s4, s5]
    weights = list(SIGNAL_WEIGHTS)
    if s6 is not None:
        signals.append(s6)
        weights.append(S6_WEIGHT)
    weighted = sum(s * w for s, w in zip(signals, weights))
    return int(100.0 * weighted / sum(weights))


async def _score_commit(
    commit: CommitModel,
    context: RegressionContext,
    diffs: dict[str, str],
    t_min: datetime,
    t_max: datetime,
    embed_fn,
) -> None:
    pr = context.prs_by_key.get(commit.pr_key)
    if pr is None:
        return
    s1 = s1_temporal_proximity(commit, t_min, t_max)
    s2 = s2_modification_intensity(diffs)
    s3 = s3_nesting_depth_shift(diffs)
    s4 = s4_control_flow_delta(diffs)
    s5 = s5_change_concentration(commit, context.files_by_key)
    s6 = (
        await s6_component_proximity(
            commit, pr, context.files_by_key, diffs, context.metric_context, embed_fn
        )
        if context.metric_context is not None
        else None
    )
    commit.triage_score = _combine_signals(s1, s2, s3, s4, s5, s6)


async def run(
    context: RegressionContext,
    flash_client: LLMClient,
    frontier_client: LLMClient,
    *,
    score_threshold: int = SCORE_THRESHOLD,
    min_passing: int = MIN_PASSING_COMMITS,
) -> RegressionContext:
    """Run the full Phase 2 pipeline. Mutates and returns context."""
    logger.info("Heuristics: discarding files")
    for file in context.files_by_key.values():
        if reason := should_discard(file):
            file.discarded = True
            file.discard_reason = reason

    for commit in context.commits_by_key.values():
        has_active = any(
            not context.files_by_key[k].discarded
            for k in commit.file_keys
            if k in context.files_by_key
        )
        if not has_active:
            commit.passed_heuristics = False

    total = len(context.commits_by_key)
    survivors = sum(1 for c in context.commits_by_key.values() if c.passed_heuristics)
    logger.info("Heuristics: %d/%d commits have surviving files", survivors, total)

    diffs_by_commit = load_all_diffs(context)

    surviving = [c for c in context.commits_by_key.values() if c.passed_heuristics]

    if surviving:
        timestamps = [c.committed_at for c in surviving if c.committed_at]
        now = datetime.now(timezone.utc)
        t_min = min(timestamps, default=now)
        t_max = max(timestamps, default=now)

        await asyncio.gather(
            *[
                _score_commit(
                    commit,
                    context,
                    diffs_by_commit.get(commit.commit_key, {}),
                    t_min,
                    t_max,
                    frontier_client.embed,
                )
                for commit in surviving
            ]
        )

        ranked = sorted(surviving, key=lambda c: c.triage_score, reverse=True)
        for i, commit in enumerate(ranked):
            if i >= min_passing and commit.triage_score < score_threshold:
                commit.passed_heuristics = False

        kept = sum(1 for c in surviving if c.passed_heuristics)
        logger.info(
            "Heuristics: %d commits above score threshold %d", kept, score_threshold
        )

    flash_commit_count = sum(
        1 for c in context.commits_by_key.values() if c.passed_heuristics
    )
    logger.info("Flash: building prompts for %d commits", flash_commit_count)
    flash_prompts = await build_flash_prompts(context, diffs_by_commit, flash_client)
    flash_keys: set[str] = set()
    for prompt in flash_prompts:
        response = await flash_client.complete_structured(prompt, FlashResponse)
        flash_keys.update(
            d.commit_key for d in response.decisions if d.worth_investigating
        )
    logger.info("Flash: %d commits flagged for deep analysis", len(flash_keys))

    logger.info("Frontier: deep analysis on %d commits", len(flash_keys))
    frontier_prompts = await build_frontier_prompts(
        context, flash_keys, diffs_by_commit, frontier_client
    )
    rankings = []
    for prompt in frontier_prompts:
        response = await frontier_client.complete_structured(prompt, FrontierResponse)
        rankings.extend(response.rankings)

    rankings.sort(key=lambda r: r.triage_score, reverse=True)
    for i, ranking in enumerate(rankings[:MAX_FRONTIER_RANKINGS]):
        if commit := context.commits_by_key.get(ranking.commit_key):
            commit.triage_score = ranking.triage_score
            commit.confidence = ranking.confidence
            commit.needs_deep_analysis = i < NEEDS_DEEP_ANALYSIS_TOP_N

    logger.info("Frontier: ranked %d commits", len(rankings))

    report_path = save_report(context, context.workspace_dir)
    logger.info("Report saved: %s", report_path)

    logger.info("Pipeline complete: culprit candidates ranked")
    return context


async def run_for_jira(
    jira_key: str,
    jira_client: "JiraClient",
    github_token: str,
    flash_client: LLMClient,
    frontier_client: LLMClient,
) -> None:
    """Fetch regression data from JIRA, run the pipeline, and upload the report."""
    from perf_keeper.commit_triage.data_acquisition import build_regression_context

    markdown = jira_client.get_markdown_attachment(jira_key)

    # TODO: parse markdown → regression fields (requires markdown parser)
    # reg = parse(markdown)
    # context = await build_regression_context(github_token=github_token, ...)
    # if reg.metric_context:
    #     context.metric_context = reg.metric_context
    # context = await run(context, flash_client, frontier_client)
    # report = (context.workspace_dir / f"commit-triage-{jira_key}.md").read_text()
    # jira_client.upload_attachment(jira_key, f"commit-triage-{jira_key}.md", report)

    raise NotImplementedError("Markdown parser required to extract regression fields")
