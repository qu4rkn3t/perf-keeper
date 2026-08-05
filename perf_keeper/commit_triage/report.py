"""Generate a Markdown commit-triage report structured for Qdrant + BM25 hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

from perf_keeper.commit_triage.models import PRModel, RegressionContext


def build_signature(context: RegressionContext) -> str:
    """Return a dense natural-language paragraph describing the regression, suitable for vector embedding."""
    pm = context.primary_metric
    parts: list[str] = []

    title = f" ({context.jira_title})" if context.jira_title else ""
    parts.append(
        f"Regression {context.jira_key}{title} detected in the {context.test} test."
    )

    value = f" (value: {pm.value:.2f})" if pm.value is not None else ""
    parts.append(
        f"The {pm.name} metric changed by {pm.change:+.2f}%{value}"
        f" in payload {context.bad_version}"
        f" compared to baseline {context.good_version}."
    )

    if len(context.all_affected_metrics) > 1:
        others = ", ".join(
            f"{m.name} ({m.change:+.1f}%)" for m in context.all_affected_metrics[1:]
        )
        parts.append(f"Additional affected metrics: {others}.")

    if context.prs_by_key:
        repos = sorted({f"{pr.owner}/{pr.repo}" for pr in context.prs_by_key.values()})
        repo_clause = ", ".join(repos[:6]) + (" and others" if len(repos) > 6 else "")
        n_pr, n_cm = len(context.prs_by_key), len(context.commits_by_key)
        parts.append(
            f"The regression window spans {n_pr} pull request{'s' if n_pr != 1 else ''}"
            f" and {n_cm} commit{'s' if n_cm != 1 else ''}"
            f" across {len(repos)} repositor{'ies' if len(repos) != 1 else 'y'}"
            f" including {repo_clause}."
        )

    if context.jira_labels:
        parts.append(f"Labels: {', '.join(context.jira_labels)}.")

    if mc := context.metric_context:
        parts.append(f"Metric description: {mc.description}.")
        if mc.related_repos:
            parts.append(f"Related repositories: {', '.join(mc.related_repos)}.")
        if mc.known_causes:
            parts.append(f"Known causes: {'; '.join(mc.known_causes)}.")

    return " ".join(parts)


def build_report(context: RegressionContext, max_rankings: int = 15) -> str:
    """Return the full Markdown report with signature, key-term index, and ranked commits."""
    pm = context.primary_metric
    lines: list[str] = []

    lines += [
        "## Regression Signature",
        "",
        build_signature(context),
        "",
    ]

    title_line = (
        f"{context.jira_key} — {context.jira_title}"
        if context.jira_title
        else context.jira_key
    )
    facts = [
        ("JIRA", title_line),
        ("Test", context.test),
        ("Primary metric", f"{pm.name} {pm.change:+.1f}%"),
        ("Good payload", context.good_version),
        ("Bad payload", context.bad_version),
        ("PRs in window", str(len(context.prs_by_key))),
        ("Commits triaged", str(len(context.commits_by_key))),
    ]
    if context.jira_labels:
        facts.append(("Labels", ", ".join(context.jira_labels)))
    lines += [f"{k}: {v}" for k, v in facts]
    lines.append("")

    repos = sorted({f"{pr.owner}/{pr.repo}" for pr in context.prs_by_key.values()})
    kw_entities: list[str] = [
        context.jira_key,
        context.test,
        pm.name,
        context.good_version,
        context.bad_version,
        *context.jira_labels,
        *repos,
    ]
    if mc := context.metric_context:
        kw_entities += mc.related_repos + mc.known_causes

    lines += [
        "## Keywords",
        "",
        "  ".join(dict.fromkeys(t for t in kw_entities if t)),
        "",
    ]

    ranked = sorted(
        (c for c in context.commits_by_key.values() if c.triage_score > 0),
        key=lambda c: c.triage_score,
        reverse=True,
    )[:max_rankings]

    if ranked:
        lines += ["---", "", "## Commit Rankings", ""]
        for i, commit in enumerate(ranked, 1):
            pr = context.prs_by_key.get(commit.pr_key)
            confidence = (
                commit.confidence.value.capitalize() if commit.confidence else "—"
            )
            top = " — Top Candidate" if i == 1 else ""
            lines.append(f"### Rank {i}{top} — {_pr_label(pr, commit.pr_key)}")
            lines.append(
                f"Score: {commit.triage_score}  Confidence: {confidence}  SHA: {commit.sha[:8]}"
            )
            if pr:
                lines.append(f"PR title: {pr.title}")
            lines.append(f"Commit: {commit.message.split(chr(10))[0]}")
            lines.append("")

    return "\n".join(lines)


def save_report(context: RegressionContext, directory: Path | str = ".") -> Path:
    """Write the report to commit-triage-{jira_key}.md and return its path."""
    out = Path(directory) / f"commit-triage-{context.jira_key}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(context), encoding="utf-8")
    return out


def _pr_label(pr: PRModel | None, fallback: str) -> str:
    if pr is None:
        return fallback
    return f"[{pr.owner}/{pr.repo}#{pr.number}](https://github.com/{pr.owner}/{pr.repo}/pull/{pr.number})"
