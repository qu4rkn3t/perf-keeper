"""Prompt building for flash and frontier LLM stages."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from perf_keeper.commit_triage.config import (
    CONTEXT_BUDGET_FRACTION,
    HUNK_HEADER_RE,
)
from perf_keeper.commit_triage.llm import LLMClient
from perf_keeper.commit_triage.models import (
    CommitModel,
    FileModel,
    PRModel,
    RegressionContext,
)

logger = logging.getLogger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"
_FLASH_TEMPLATE = (_TEMPLATES / "flash.md").read_text(encoding="utf-8")
_FRONTIER_TEMPLATE = (_TEMPLATES / "frontier.md").read_text(encoding="utf-8")

_REGRESSION_HEADER = """\
## Regression
JIRA: {jira_key} — {jira_title}
Test: {test}
Primary metric: {metric_name} {change:+.1f}% (value: {value})
Affected metrics: {all_metrics}
Good payload: {good_version}
Bad payload: {bad_version}"""

_METRIC_CONTEXT_BLOCK = """\

## Metric Context
Name: {name}
Description: {description}
Related repositories: {repos}
Known causes: {causes}"""


def load_all_diffs(context: RegressionContext) -> dict[str, dict[str, str]]:
    """Load non-discarded diffs for all commits. Returns {commit_key: {file_path: diff_text}}."""
    result: dict[str, dict[str, str]] = {}
    for commit_key, commit in context.commits_by_key.items():
        diffs: dict[str, str] = {}
        for file_key in commit.file_keys:
            f = context.files_by_key.get(file_key)
            if f and not f.discarded and f.diff_path.exists():
                diffs[f.path] = f.diff_path.read_text(
                    encoding="utf-8", errors="replace"
                )
        if diffs:
            result[commit_key] = diffs
    return result


def _extract_functions(diff_text: str) -> list[str]:
    """Return unique function names from @@ hunk headers in order of appearance."""
    seen: set[str] = set()
    funcs: list[str] = []
    for line in diff_text.splitlines():
        m = HUNK_HEADER_RE.match(line)
        if m and (fn := (m.group(1) or "").strip()):
            if fn not in seen:
                seen.add(fn)
                funcs.append(fn)
    return funcs


def _regression_header(context: RegressionContext) -> str:
    pm = context.primary_metric
    all_metrics = (
        ", ".join(f"{m.name} {m.change:+.1f}%" for m in context.all_affected_metrics)
        or "—"
    )
    header = _REGRESSION_HEADER.format(
        jira_key=context.jira_key,
        jira_title=context.jira_title or "—",
        test=context.test,
        metric_name=pm.name,
        change=pm.change,
        value=f"{pm.value:.2f}" if pm.value is not None else "—",
        all_metrics=all_metrics,
        good_version=context.good_version,
        bad_version=context.bad_version,
    )
    if mc := context.metric_context:
        header += _METRIC_CONTEXT_BLOCK.format(
            name=mc.name,
            description=mc.description,
            repos=", ".join(mc.related_repos) or "—",
            causes="; ".join(mc.known_causes) or "—",
        )
    return header


def _flash_commit_block(commit: CommitModel, diffs: dict[str, str]) -> str:
    ts = commit.committed_at.isoformat() if commit.committed_at else "unknown"
    first_line = commit.message.split("\n")[0]
    lines = [f"#### [{commit.commit_key}] {ts}", first_line, "", "Files:"]
    for path, diff_text in diffs.items():
        fns = _extract_functions(diff_text)
        lines.append(f"  {path}  →  {', '.join(fns) if fns else '—'}")
    return "\n".join(lines)


def _flash_pr_block(
    pr: PRModel,
    context: RegressionContext,
    diffs_by_commit: dict[str, dict[str, str]],
) -> tuple[str, list[str]]:
    """Return (block_text, commit_keys) or ('', []) when no passing commits exist."""
    commit_blocks: list[str] = []
    commit_keys: list[str] = []

    for commit_key in pr.commit_keys:
        commit = context.commits_by_key.get(commit_key)
        if not commit or not commit.passed_heuristics:
            continue
        diffs = diffs_by_commit.get(commit_key, {})
        if not diffs:
            continue
        commit_blocks.append(_flash_commit_block(commit, diffs))
        commit_keys.append(commit_key)

    if not commit_blocks:
        return "", []

    labels = ", ".join(pr.labels) if pr.labels else "none"
    text = f"### {pr.pr_key}: {pr.title}\nLabels: {labels}\n\n" + "\n\n".join(
        commit_blocks
    )
    return text, commit_keys


def _frontier_file_block(f: FileModel, diff_text: str) -> str:
    lines = [f"##### {f.path}"]
    if f.review_comments:
        lines.append("Review comments:")
        for rc in f.review_comments:
            ref = f"line {rc.line}" if rc.line else "general"
            lines.append(f"  [{ref}] {rc.body}")
        lines.append("")
    if diff_text:
        lines += ["```diff", diff_text, "```"]
    return "\n".join(lines)


def _frontier_commit_block(
    commit: CommitModel,
    files_by_key: dict[str, FileModel],
    diffs: dict[str, str],
) -> str:
    ts = commit.committed_at.isoformat() if commit.committed_at else "unknown"
    lines = [f"#### [{commit.commit_key}] {ts}", commit.message, ""]
    for file_key in commit.file_keys:
        f = files_by_key.get(file_key)
        if not f or f.discarded:
            continue
        lines.append(_frontier_file_block(f, diffs.get(f.path, "")))
        lines.append("")
    return "\n".join(lines)


def _frontier_pr_block(
    pr: PRModel,
    context: RegressionContext,
    diffs_by_commit: dict[str, dict[str, str]],
    commit_keys: set[str],
) -> tuple[str, list[str]]:
    """Return (block_text, commit_keys) or ('', []) when no included commits exist."""
    commit_blocks: list[str] = []
    included_keys: list[str] = []

    for ck in pr.commit_keys:
        if ck not in commit_keys:
            continue
        commit = context.commits_by_key.get(ck)
        if not commit:
            continue
        diffs = diffs_by_commit.get(ck, {})
        commit_blocks.append(
            _frontier_commit_block(commit, context.files_by_key, diffs)
        )
        included_keys.append(ck)

    if not commit_blocks:
        return "", []

    labels = ", ".join(pr.labels) if pr.labels else "none"
    sections: list[str] = [f"### {pr.pr_key}: {pr.title}", f"Labels: {labels}"]
    if pr.description:
        sections.append(f"Description: {pr.description}")
    if pr.comments:
        sections.append("PR comments:")
        for c in pr.comments:
            sections.append(f"  • {c.body}")
    if pr.review_summaries:
        sections.append("Reviews:")
        for r in pr.review_summaries:
            state = r.state.value if r.state else "—"
            sections.append(f"  [{state}] {r.body}")
    sections.append("")
    sections.extend(commit_blocks)
    return "\n".join(sections), included_keys


async def _split_prompts(
    header: str,
    footer_template: str,
    pr_blocks: list[tuple[str, list[str]]],
    client: LLMClient,
    fraction: float,
) -> list[str]:
    budget = client.token_budget(fraction)

    texts = [header, footer_template.replace("{commit_keys}", "")] + [
        block + ", ".join(keys) for block, keys in pr_blocks
    ]
    counts = list(await asyncio.gather(*[client.count_tokens(t) for t in texts]))
    header_tokens, footer_base = counts[0], counts[1]
    block_costs = counts[2:]

    def assemble(blocks: list[tuple[str, list[str]]]) -> str:
        keys = ", ".join(k for _, ks in blocks for k in ks)
        body = "\n\n".join(b for b, _ in blocks)
        return f"{header}\n\n{body}\n\n" + footer_template.replace(
            "{commit_keys}", keys
        )

    prompts: list[str] = []
    current: list[tuple[str, list[str]]] = []
    current_tokens = header_tokens + footer_base

    for (block, keys), cost in zip(pr_blocks, block_costs):
        if current_tokens + cost > budget and current:
            prompts.append(assemble(current))
            current = [(block, keys)]
            current_tokens = header_tokens + footer_base + cost
        else:
            current.append((block, keys))
            current_tokens += cost

    if current:
        prompts.append(assemble(current))
    result = prompts or [assemble([])]
    logger.debug(
        "Split %d PR blocks into %d prompt(s) (budget: %d tokens)",
        len(pr_blocks),
        len(result),
        budget,
    )
    return result


def _header_footer(template: str, regression: str) -> tuple[str, str]:
    """Split a template on {pr_blocks} into (header, footer_template)."""
    parts = template.split("{pr_blocks}", maxsplit=1)
    return parts[0].replace("{regression}", regression), parts[1]


async def build_flash_prompts(
    context: RegressionContext,
    diffs_by_commit: dict[str, dict[str, str]],
    client: LLMClient,
    fraction: float = CONTEXT_BUDGET_FRACTION,
) -> list[str]:
    """Build one or more flash prompts covering all commits that passed heuristics."""
    regression = _regression_header(context)
    header, footer_tpl = _header_footer(_FLASH_TEMPLATE, regression)
    pr_blocks = [
        _flash_pr_block(pr, context, diffs_by_commit)
        for pr in context.prs_by_key.values()
    ]
    filtered = [(b, ks) for b, ks in pr_blocks if b]
    commit_count = sum(len(ks) for _, ks in filtered)
    prompts = await _split_prompts(header, footer_tpl, filtered, client, fraction)
    logger.info("Flash: built %d prompt(s) for %d commits", len(prompts), commit_count)
    return prompts


async def build_frontier_prompts(
    context: RegressionContext,
    commit_keys: set[str],
    diffs_by_commit: dict[str, dict[str, str]],
    client: LLMClient,
    fraction: float = CONTEXT_BUDGET_FRACTION,
) -> list[str]:
    """Build one or more frontier prompts covering commits that passed flash triage."""
    regression = _regression_header(context)
    header, footer_tpl = _header_footer(_FRONTIER_TEMPLATE, regression)
    pr_blocks = [
        _frontier_pr_block(pr, context, diffs_by_commit, commit_keys)
        for pr in context.prs_by_key.values()
    ]
    filtered = [(b, ks) for b, ks in pr_blocks if b]
    commit_count = sum(len(ks) for _, ks in filtered)
    prompts = await _split_prompts(header, footer_tpl, filtered, client, fraction)
    logger.info(
        "Frontier: built %d prompt(s) for %d commits", len(prompts), commit_count
    )
    return prompts
