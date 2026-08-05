"""Data acquisition: Build RegressionContext from GitHub PR data."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from perf_keeper.commit_triage.config import (
    DEFAULT_CONCURRENCY,
    GITHUB_API,
    HTTP_TIMEOUT,
    MAX_RETRIES,
    PER_PAGE,
    PR_URL_RE,
)
from perf_keeper.commit_triage.models import (
    AffectedMetric,
    CommitModel,
    FileModel,
    FileStatus,
    PRComment,
    PRModel,
    RegressionContext,
    ReviewComment,
    ReviewSummary,
    ReviewStatus,
)

logger = logging.getLogger(__name__)


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Return (owner, repo, number) from a GitHub PR URL."""
    m = PR_URL_RE.search(url)
    if not m:
        raise ValueError(f"Cannot parse GitHub PR URL: {url!r}")
    return m.group(1), m.group(2), int(m.group(3))


def make_affected_metric(
    name: str, change: float, value: float | None = None
) -> AffectedMetric:
    """Convenience constructor for AffectedMetric."""
    return AffectedMetric(name=name, change=change, value=value)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _file_status(raw: str) -> FileStatus | None:
    try:
        return FileStatus(raw)
    except ValueError:
        return None


def _review_status(raw: str) -> ReviewStatus | None:
    try:
        return ReviewStatus(raw)
    except ValueError:
        return None


def _diff_path(workspace_dir: Path, sha: str, file_path: str) -> Path:
    name = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    return workspace_dir / "diffs" / sha[:8] / f"{name}.diff"


def _next_link(link_header: str) -> str:
    for segment in link_header.split(","):
        if 'rel="next"' in segment:
            m = re.search(r"<([^>]+)>", segment)
            if m:
                return m.group(1)
    return ""


class _GitHubFetcher:
    """Async GitHub API client with concurrency control and automatic retries."""

    def __init__(
        self, client: httpx.AsyncClient, concurrency: int = DEFAULT_CONCURRENCY
    ) -> None:
        self._client = client
        self._sem = asyncio.Semaphore(concurrency)

    async def _request(
        self, url: str, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        resp: httpx.Response | None = None
        last_exc: Exception | None = None
        retry_after = 0.0
        for attempt in range(MAX_RETRIES):
            if retry_after:
                await asyncio.sleep(retry_after)
            try:
                async with self._sem:
                    logger.debug("GET %s", url)
                    resp = await self._client.get(url, params=params)
            except (httpx.ReadError, httpx.ConnectError) as exc:
                last_exc = exc
                retry_after = float(2 ** (attempt + 1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                logger.warning(
                    "GitHub %d for %s, retrying (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                )
                continue
            resp.raise_for_status()
            return resp
        if resp is None:
            raise last_exc or RuntimeError(f"All {MAX_RETRIES} retries failed for {url}")
        resp.raise_for_status()
        raise RuntimeError("unreachable")

    async def get(self, url: str) -> Any:
        return (await self._request(url)).json()

    async def paginate(self, url: str) -> list[dict[str, Any]]:
        """Fetch all pages of a GitHub API endpoint."""
        items: list[dict[str, Any]] = []
        next_url = url
        params: dict[str, Any] | None = {"per_page": PER_PAGE}
        while next_url:
            resp = await self._request(next_url, params)
            items.extend(resp.json())
            next_url = _next_link(resp.headers.get("link", ""))
            params = None
        return items

    async def fetch_commit(
        self,
        owner: str,
        repo: str,
        sha: str,
        pr_key: str,
        workspace_dir: Path,
    ) -> tuple[CommitModel, list[FileModel]]:
        """Fetch a commit and build its CommitModel and FileModels."""
        logger.info("Fetching commit %s from %s/%s", sha[:8], owner, repo)
        try:
            data = await self.get(f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404):
                logger.warning(
                    "Commit %s in %s/%s inaccessible: HTTP %d",
                    sha[:8],
                    owner,
                    repo,
                    exc.response.status_code,
                )
                commit_key = f"{pr_key}:{sha}"
                return CommitModel(
                    commit_key=commit_key,
                    sha=sha,
                    message="[inaccessible]",
                    pr_key=pr_key,
                ), []
            raise
        commit_key = f"{pr_key}:{sha}"
        message: str = data["commit"]["message"]
        committed_at = _parse_dt(data["commit"]["committer"].get("date"))

        files: list[FileModel] = []
        for f in data.get("files", []):
            path: str = f["filename"]
            file_key = f"{commit_key}:{path}"
            patch: str = f.get("patch", "")
            diff_p = _diff_path(workspace_dir, sha, path)
            if patch:
                diff_p.parent.mkdir(parents=True, exist_ok=True)
                diff_p.write_text(patch, encoding="utf-8")
            files.append(
                FileModel(
                    file_key=file_key,
                    commit_key=commit_key,
                    path=path,
                    diff_path=diff_p,
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    status=_file_status(f.get("status", "")),
                )
            )

        commit = CommitModel(
            commit_key=commit_key,
            sha=sha,
            message=message,
            pr_key=pr_key,
            committed_at=committed_at,
            file_keys=[f.file_key for f in files],
            total_additions=sum(f.additions for f in files),
            total_deletions=sum(f.deletions for f in files),
            total_files_changed=len(files),
        )
        return commit, files

    async def fetch_pr(
        self,
        owner: str,
        repo: str,
        number: int,
        workspace_dir: Path,
    ) -> tuple[PRModel, list[CommitModel], list[FileModel]]:
        """Fetch a PR and build its PRModel, CommitModels, and FileModels."""
        logger.info("Fetching PR %s#%d", owner + "/" + repo, number)
        base = f"{GITHUB_API}/repos/{owner}/{repo}"
        pr_key = f"{owner}/{repo}#{number}"

        try:
            (
                pr_data,
                raw_commits,
                raw_reviews,
                raw_issue_comments,
                raw_review_comments,
            ) = await asyncio.gather(
                self.get(f"{base}/pulls/{number}"),
                self.paginate(f"{base}/pulls/{number}/commits"),
                self.paginate(f"{base}/pulls/{number}/reviews"),
                self.paginate(f"{base}/issues/{number}/comments"),
                self.paginate(f"{base}/pulls/{number}/comments"),
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404):
                logger.warning(
                    "PR %s#%d inaccessible: HTTP %d",
                    owner + "/" + repo,
                    number,
                    exc.response.status_code,
                )
                return (
                    PRModel(
                        pr_key=pr_key,
                        owner=owner,
                        repo=repo,
                        number=number,
                        title="[inaccessible]",
                    ),
                    [],
                    [],
                )
            raise

        rc_lookup: dict[tuple[str, str], list[ReviewComment]] = {}
        for rc in raw_review_comments:
            key = (
                rc.get("original_commit_id") or rc.get("commit_id", ""),
                rc.get("path", ""),
            )
            rc_lookup.setdefault(key, []).append(
                ReviewComment(
                    body=rc.get("body", ""),
                    line=rc.get("line") or rc.get("original_line"),
                    created_at=_parse_dt(rc.get("created_at")),
                )
            )

        commit_results: list[tuple[CommitModel, list[FileModel]]] = list(
            await asyncio.gather(
                *[
                    self.fetch_commit(owner, repo, c["sha"], pr_key, workspace_dir)
                    for c in raw_commits
                ]
            )
        )

        all_commits: list[CommitModel] = []
        all_files: list[FileModel] = []
        commit_keys: list[str] = []

        for commit, files in commit_results:
            for f in files:
                if comments := rc_lookup.get((commit.sha, f.path)):
                    f.review_comments = comments
            all_commits.append(commit)
            all_files.extend(files)
            commit_keys.append(commit.commit_key)

        pr = PRModel(
            pr_key=pr_key,
            owner=owner,
            repo=repo,
            number=number,
            title=pr_data.get("title", ""),
            description=pr_data.get("body") or "",
            commit_keys=commit_keys,
            labels=[lbl["name"] for lbl in pr_data.get("labels", [])],
            comments=[
                PRComment(
                    body=c.get("body", ""),
                    created_at=_parse_dt(c.get("created_at")),
                )
                for c in raw_issue_comments
            ],
            review_summaries=[
                ReviewSummary(
                    state=_review_status(r.get("state", "")),
                    body=r.get("body") or "",
                    submitted_at=_parse_dt(r.get("submitted_at")),
                )
                for r in raw_reviews
            ],
            total_additions=sum(c.total_additions for c in all_commits),
            total_deletions=sum(c.total_deletions for c in all_commits),
            total_files_changed=sum(c.total_files_changed for c in all_commits),
        )
        return pr, all_commits, all_files


async def build_regression_context(
    *,
    github_token: str,
    jira_key: str,
    jira_title: str = "",
    jira_labels: list[str],
    test: str,
    good_version: str,
    bad_version: str,
    pr_urls: list[str],
    primary_metric: AffectedMetric,
    all_affected_metrics: list[AffectedMetric] | None = None,
    perf_keeper_report_path: Path,
    workspace_dir: Path | None = None,
    analysis_id: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> RegressionContext:
    """Fetch all PR/commit/file data from GitHub and assemble a RegressionContext."""
    aid = analysis_id or str(uuid.uuid4())
    wdir = workspace_dir or Path(f"/tmp/perf-keeper/{aid}")
    wdir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    parsed_prs = [parse_pr_url(url) for url in pr_urls]

    logger.info("Starting GitHub fetch: %d PRs", len(pr_urls))
    t0 = time.monotonic()
    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as client:
        fetcher = _GitHubFetcher(client, concurrency=concurrency)
        results: list[tuple[PRModel, list[CommitModel], list[FileModel]]] = list(
            await asyncio.gather(
                *[
                    fetcher.fetch_pr(owner, repo, number, wdir)
                    for owner, repo, number in parsed_prs
                ]
            )
        )

    prs_by_key: dict[str, PRModel] = {}
    commits_by_key: dict[str, CommitModel] = {}
    files_by_key: dict[str, FileModel] = {}

    for pr, commits, files in results:
        prs_by_key[pr.pr_key] = pr
        for commit in commits:
            commits_by_key[commit.commit_key] = commit
        for f in files:
            files_by_key[f.file_key] = f

    logger.info(
        "GitHub fetch done: %d PRs, %d commits, %d files in %.1fs",
        len(prs_by_key),
        len(commits_by_key),
        len(files_by_key),
        time.monotonic() - t0,
    )
    return RegressionContext(
        analysis_id=aid,
        workspace_dir=wdir,
        jira_key=jira_key,
        jira_title=jira_title,
        jira_labels=jira_labels,
        test=test,
        primary_metric=primary_metric,
        all_affected_metrics=all_affected_metrics or [],
        good_version=good_version,
        bad_version=bad_version,
        perf_keeper_report_path=perf_keeper_report_path,
        prs_by_key=prs_by_key,
        commits_by_key=commits_by_key,
        files_by_key=files_by_key,
    )
