"""Tests for GitHub data acquisition."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx

from perf_keeper.commit_triage.data_acquisition import (
    _GitHubFetcher,
    _diff_path,
    _file_status,
    _next_link,
    _parse_dt,
    _review_status,
    make_affected_metric,
    parse_pr_url,
)
from perf_keeper.commit_triage.models import FileStatus, ReviewStatus


def _mock_response(data, status: int = 200, link: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.headers = MagicMock()
    resp.headers.get.side_effect = lambda key, default=None: (
        link if key == "link" else default
    )
    resp.raise_for_status = MagicMock()
    return resp


# parse_pr_url


class TestParsePrUrl:
    def test_standard_https_url(self):
        owner, repo, num = parse_pr_url(
            "https://github.com/openshift/kubernetes/pull/1234"
        )
        assert owner == "openshift"
        assert repo == "kubernetes"
        assert num == 1234

    def test_url_with_trailing_path(self):
        owner, repo, num = parse_pr_url("https://github.com/org/repo/pull/99/files")
        assert owner == "org"
        assert repo == "repo"
        assert num == 99

    def test_non_github_url_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_pr_url("https://gitlab.com/org/repo/merge_requests/1")

    def test_plain_string_raises(self):
        with pytest.raises(ValueError):
            parse_pr_url("openshift/kubernetes#1234")


# make_affected_metric


class TestMakeAffectedMetric:
    def test_without_value(self):
        m = make_affected_metric("cpu", -10.0)
        assert m.name == "cpu"
        assert m.change == -10.0
        assert m.value is None

    def test_with_value(self):
        m = make_affected_metric("latency", 15.0, 250.5)
        assert m.value == 250.5


# _parse_dt


class TestParseDt:
    def test_none_returns_none(self):
        assert _parse_dt(None) is None

    def test_empty_returns_none(self):
        assert _parse_dt("") is None

    def test_z_suffix(self):
        dt = _parse_dt("2025-01-15T12:30:00Z")
        assert dt is not None
        assert dt.year == 2025 and dt.month == 1 and dt.day == 15

    def test_offset_suffix(self):
        dt = _parse_dt("2025-06-01T00:00:00+00:00")
        assert dt is not None
        assert dt.month == 6


# _file_status


class TestFileStatus:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("added", FileStatus.ADDED),
            ("modified", FileStatus.MODIFIED),
            ("removed", FileStatus.REMOVED),
            ("renamed", FileStatus.RENAMED),
        ],
    )
    def test_known_statuses(self, raw, expected):
        assert _file_status(raw) == expected

    def test_unknown_returns_none(self):
        assert _file_status("changed") is None


# _review_status


class TestReviewStatus:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("APPROVED", ReviewStatus.APPROVED),
            ("CHANGES_REQUESTED", ReviewStatus.CHANGES_REQUESTED),
            ("COMMENTED", ReviewStatus.COMMENTED),
            ("DISMISSED", ReviewStatus.DISMISSED),
        ],
    )
    def test_known_statuses(self, raw, expected):
        assert _review_status(raw) == expected

    def test_unknown_returns_none(self):
        assert _review_status("MERGED") is None


# _next_link


class TestNextLink:
    def test_empty_header(self):
        assert _next_link("") == ""

    def test_no_next_rel(self):
        assert _next_link('<url>; rel="prev"') == ""

    def test_extracts_next_url(self):
        header = '<https://api.github.com/repos/org/repo/pulls/1/commits?page=2>; rel="next", <...>; rel="last"'
        assert (
            _next_link(header)
            == "https://api.github.com/repos/org/repo/pulls/1/commits?page=2"
        )

    def test_next_with_no_other_rels(self):
        header = '<https://api.github.com/page/2>; rel="next"'
        assert _next_link(header) == "https://api.github.com/page/2"


# _diff_path


class TestDiffPath:
    def test_uses_first_eight_chars_of_sha(self, tmp_path):
        p = _diff_path(tmp_path, "abcdef1234567890", "file.go")
        assert p.parent.name == "abcdef12"

    def test_is_under_diffs_dir(self, tmp_path):
        p = _diff_path(tmp_path, "sha", "file.go")
        assert p.parent.parent.name == "diffs"

    def test_has_diff_extension(self, tmp_path):
        p = _diff_path(tmp_path, "sha", "file.go")
        assert p.suffix == ".diff"

    def test_same_inputs_same_path(self, tmp_path):
        assert _diff_path(tmp_path, "sha123", "a.go") == _diff_path(
            tmp_path, "sha123", "a.go"
        )

    def test_different_files_different_paths(self, tmp_path):
        assert _diff_path(tmp_path, "sha", "a.go") != _diff_path(
            tmp_path, "sha", "b.go"
        )


# _GitHubFetcher.fetch_commit


class TestFetchCommit:
    def _make_commit_data(
        self, filename="server.go", patch="@@ -1 +1 @@ f\n-old\n+new\n"
    ):
        return {
            "commit": {
                "message": "Fix pod scheduling",
                "committer": {"date": "2025-01-15T12:00:00Z"},
            },
            "files": [
                {
                    "filename": filename,
                    "status": "modified",
                    "additions": 10,
                    "deletions": 5,
                    "patch": patch,
                }
            ],
        }

    async def test_builds_commit_model(self, tmp_path):
        resp = _mock_response(self._make_commit_data())
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        fetcher = _GitHubFetcher(client)

        commit, files = await fetcher.fetch_commit(
            "openshift", "kubernetes", "abc123", "openshift/kubernetes#1", tmp_path
        )

        assert commit.sha == "abc123"
        assert commit.message == "Fix pod scheduling"
        assert commit.pr_key == "openshift/kubernetes#1"
        assert commit.total_additions == 10
        assert commit.total_deletions == 5
        assert commit.total_files_changed == 1
        assert commit.committed_at is not None

    async def test_builds_file_models(self, tmp_path):
        resp = _mock_response(self._make_commit_data())
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        fetcher = _GitHubFetcher(client)

        _, files = await fetcher.fetch_commit(
            "org", "repo", "sha", "org/repo#1", tmp_path
        )

        assert len(files) == 1
        assert files[0].path == "server.go"
        assert files[0].additions == 10
        assert files[0].deletions == 5

    async def test_patch_written_to_disk(self, tmp_path):
        patch = "@@ -1 +1 @@ func\n-old\n+new\n"
        resp = _mock_response(self._make_commit_data(patch=patch))
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        fetcher = _GitHubFetcher(client)

        _, files = await fetcher.fetch_commit(
            "org", "repo", "sha123", "org/repo#1", tmp_path
        )

        assert files[0].diff_path.exists()
        assert files[0].diff_path.read_text() == patch

    async def test_binary_file_no_patch_no_diff_file(self, tmp_path):
        data = {
            "commit": {
                "message": "add binary",
                "committer": {"date": "2025-01-01T00:00:00Z"},
            },
            "files": [
                {
                    "filename": "image.png",
                    "status": "added",
                    "additions": 0,
                    "deletions": 0,
                }
            ],
        }
        resp = _mock_response(data)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        fetcher = _GitHubFetcher(client)

        _, files = await fetcher.fetch_commit(
            "org", "repo", "sha", "org/repo#1", tmp_path
        )

        assert not files[0].diff_path.exists()

    async def test_commit_key_composite(self, tmp_path):
        resp = _mock_response(self._make_commit_data())
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        fetcher = _GitHubFetcher(client)

        commit, files = await fetcher.fetch_commit(
            "org", "repo", "sha456", "org/repo#2", tmp_path
        )

        assert commit.commit_key == "org/repo#2:sha456"
        assert files[0].commit_key == "org/repo#2:sha456"

    async def test_retries_on_server_error(self, tmp_path):
        error_resp = _mock_response({}, status=500)
        ok_data = {
            "commit": {"message": "fix", "committer": {"date": "2025-01-01T00:00:00Z"}},
            "files": [],
        }
        ok_resp = _mock_response(ok_data)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=[error_resp, error_resp, ok_resp])
        fetcher = _GitHubFetcher(client)

        commit, _ = await fetcher.fetch_commit(
            "org", "repo", "sha", "org/repo#1", tmp_path
        )
        assert commit.message == "fix"
        assert client.get.call_count == 3


# _GitHubFetcher.paginate


class TestPaginate:
    async def test_single_page(self):
        items = [{"id": 1}, {"id": 2}]
        resp = _mock_response(items)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        fetcher = _GitHubFetcher(client)
        result = await fetcher.paginate("https://api.github.com/test")
        assert result == items

    async def test_follows_next_link(self):
        page1 = [{"id": 1}]
        page2 = [{"id": 2}]
        resp1 = _mock_response(
            page1, link='<https://api.github.com/test?page=2>; rel="next"'
        )
        resp1.headers.get.return_value = (
            '<https://api.github.com/test?page=2>; rel="next"'
        )
        resp2 = _mock_response(page2, link="")
        resp2.headers.get.return_value = ""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=[resp1, resp2])
        fetcher = _GitHubFetcher(client)
        result = await fetcher.paginate("https://api.github.com/test")
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
