"""Tests for prompt building."""

from __future__ import annotations


from perf_keeper.commit_triage.models import (
    CommitModel,
    FileModel,
    RegressionContext,
    ReviewComment,
)
from perf_keeper.commit_triage.prompts import (
    _extract_functions,
    _flash_commit_block,
    _flash_pr_block,
    _frontier_file_block,
    _frontier_pr_block,
    _regression_header,
    _split_prompts,
    build_flash_prompts,
    build_frontier_prompts,
    load_all_diffs,
)
from perf_keeper.commit_triage.tests.conftest import (
    COMMIT_KEY,
    FILE_KEY,
    FILE_PATH,
    PR_KEY,
    MockLLMClient,
)


# load_all_diffs


class TestLoadAllDiffs:
    def test_loads_non_discarded_with_existing_diff(self, regression_context):
        diffs = load_all_diffs(regression_context)
        assert COMMIT_KEY in diffs
        assert FILE_PATH in diffs[COMMIT_KEY]

    def test_excludes_discarded_files(self, regression_context):
        regression_context.files_by_key[FILE_KEY].discarded = True
        diffs = load_all_diffs(regression_context)
        assert diffs == {}

    def test_excludes_missing_diff_files(self, tmp_path, primary_metric):
        file_key = "pr:sha:main.go"
        f = FileModel(
            file_key=file_key,
            commit_key="pr:sha",
            path="main.go",
            diff_path=tmp_path / "nonexistent.diff",
        )
        commit = CommitModel(
            commit_key="pr:sha",
            sha="sha",
            message="m",
            pr_key="pr",
            file_keys=[file_key],
        )
        ctx = RegressionContext(
            analysis_id="t",
            workspace_dir=tmp_path,
            jira_key="P-1",
            test="t",
            primary_metric=primary_metric,
            good_version="4.17",
            bad_version="4.18",
            perf_keeper_report_path=tmp_path / "r.md",
            commits_by_key={"pr:sha": commit},
            files_by_key={file_key: f},
        )
        assert load_all_diffs(ctx) == {}


# _extract_functions


class TestExtractFunctions:
    def test_empty_diff(self):
        assert _extract_functions("") == []

    def test_no_hunk_header(self):
        assert _extract_functions("plain text\n+added\n") == []

    def test_single_function(self):
        diff = "@@ -1,2 +1,3 @@ func process() {\n-old\n+new\n"
        assert _extract_functions(diff) == ["func process() {"]

    def test_multiple_different_functions(self):
        diff = "@@ -1 +1 @@ func_a\n-x\n+y\n@@ -10 +10 @@ func_b\n-a\n+b\n"
        fns = _extract_functions(diff)
        assert fns == ["func_a", "func_b"]

    def test_duplicate_functions_deduplicated(self):
        diff = "@@ -1 +1 @@ myfunc\n-x\n+y\n@@ -10 +10 @@ myfunc\n-a\n+b\n"
        fns = _extract_functions(diff)
        assert fns == ["myfunc"]

    def test_hunk_without_function_excluded(self):
        diff = "@@ -1,1 +1,1 @@\n-old\n+new\n"
        assert _extract_functions(diff) == []


# _regression_header


class TestRegressionHeader:
    def test_contains_jira_key(self, regression_context):
        header = _regression_header(regression_context)
        assert "PERFSCALE-1234" in header

    def test_contains_jira_title(self, regression_context):
        header = _regression_header(regression_context)
        assert "Pod ready latency" in header

    def test_contains_metric_name_and_change(self, regression_context):
        header = _regression_header(regression_context)
        assert "podReadyLatency" in header
        assert "-15.0%" in header

    def test_contains_versions(self, regression_context):
        header = _regression_header(regression_context)
        assert "4.17.0" in header
        assert "4.18.0" in header

    def test_contains_test_name(self, regression_context):
        header = _regression_header(regression_context)
        assert "node-density" in header

    def test_no_jira_title_shows_dash(self, regression_context):
        regression_context.jira_title = ""
        header = _regression_header(regression_context)
        assert "—" in header

    def test_all_affected_metrics_listed(self, regression_context):
        header = _regression_header(regression_context)
        assert "apiserverCPU" in header


# block builders


class TestFlashCommitBlock:
    def test_contains_commit_key(self, regression_context):
        commit = regression_context.commits_by_key[COMMIT_KEY]
        diffs = {FILE_PATH: "@@ -1 +1 @@ syncPod\n-old\n+new\n"}
        block = _flash_commit_block(commit, diffs)
        assert COMMIT_KEY in block

    def test_contains_timestamp(self, regression_context):
        commit = regression_context.commits_by_key[COMMIT_KEY]
        diffs = {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}
        block = _flash_commit_block(commit, diffs)
        assert "2025-01-15" in block

    def test_contains_first_line_of_message(self, regression_context):
        commit = regression_context.commits_by_key[COMMIT_KEY]
        diffs = {}
        block = _flash_commit_block(commit, diffs)
        assert "Optimize pod sync" in block

    def test_contains_file_paths(self, regression_context):
        commit = regression_context.commits_by_key[COMMIT_KEY]
        diffs = {FILE_PATH: "@@ -1 +1 @@ syncPod\n-x\n+y\n"}
        block = _flash_commit_block(commit, diffs)
        assert FILE_PATH in block

    def test_contains_extracted_functions(self, regression_context):
        commit = regression_context.commits_by_key[COMMIT_KEY]
        diffs = {FILE_PATH: "@@ -1 +1 @@ func syncPod() error {\n-x\n+y\n"}
        block = _flash_commit_block(commit, diffs)
        assert "func syncPod() error {" in block

    def test_no_committed_at_shows_unknown(self):
        commit = CommitModel(commit_key="p:s", sha="s", message="fix", pr_key="p")
        block = _flash_commit_block(commit, {})
        assert "unknown" in block


class TestFlashPrBlock:
    def test_returns_empty_when_no_passing_commits(self, regression_context):
        regression_context.commits_by_key[COMMIT_KEY].passed_heuristics = False
        block, keys = _flash_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
        )
        assert block == ""
        assert keys == []

    def test_returns_empty_when_no_diffs(self, regression_context):
        block, keys = _flash_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
        )
        assert block == ""
        assert keys == []

    def test_returns_block_with_diffs(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        block, keys = _flash_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            diffs,
        )
        assert block != ""
        assert COMMIT_KEY in keys

    def test_block_contains_pr_title(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        block, _ = _flash_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            diffs,
        )
        assert "Optimize pod scheduling" in block

    def test_block_contains_labels(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        block, _ = _flash_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            diffs,
        )
        assert "performance" in block


class TestFrontierFileBlock:
    def test_contains_path(self, regression_context):
        f = regression_context.files_by_key[FILE_KEY]
        block = _frontier_file_block(f, "@@ -1 +1 @@ f\n-x\n+y\n")
        assert FILE_PATH in block

    def test_contains_diff_in_code_fence(self, regression_context):
        diff = "@@ -1 +1 @@ f\n-x\n+y\n"
        f = regression_context.files_by_key[FILE_KEY]
        block = _frontier_file_block(f, diff)
        assert "```diff" in block
        assert diff in block

    def test_empty_diff_no_code_fence(self, regression_context):
        f = regression_context.files_by_key[FILE_KEY]
        block = _frontier_file_block(f, "")
        assert "```diff" not in block

    def test_review_comments_included(self, regression_context):
        f = regression_context.files_by_key[FILE_KEY]
        f.review_comments = [ReviewComment(body="Fix this!", line=10)]
        block = _frontier_file_block(f, "")
        assert "Fix this!" in block
        assert "line 10" in block

    def test_review_comment_without_line(self, regression_context):
        f = regression_context.files_by_key[FILE_KEY]
        f.review_comments = [ReviewComment(body="General comment")]
        block = _frontier_file_block(f, "")
        assert "general" in block


class TestFrontierPrBlock:
    def test_returns_empty_when_no_included_commits(self, regression_context):
        block, keys = _frontier_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
            set(),
        )
        assert block == ""
        assert keys == []

    def test_returns_block_for_included_commits(self, regression_context):
        block, keys = _frontier_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
            {COMMIT_KEY},
        )
        assert block != ""
        assert COMMIT_KEY in keys

    def test_includes_pr_description(self, regression_context):
        block, _ = _frontier_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
            {COMMIT_KEY},
        )
        assert "allocations" in block

    def test_includes_pr_comments(self, regression_context):
        block, _ = _frontier_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
            {COMMIT_KEY},
        )
        assert "LGTM" in block

    def test_includes_review_summaries(self, regression_context):
        block, _ = _frontier_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
            {COMMIT_KEY},
        )
        assert "APPROVED" in block

    def test_full_description_included(self, regression_context):
        regression_context.prs_by_key[PR_KEY].description = "x" * 5000
        block, _ = _frontier_pr_block(
            regression_context.prs_by_key[PR_KEY],
            regression_context,
            {},
            {COMMIT_KEY},
        )
        assert "x" * 5000 in block


# _split_prompts


class TestSplitPrompts:
    async def test_empty_blocks_single_prompt(self):
        client = MockLLMClient()
        prompts = await _split_prompts(
            "header", "footer {commit_keys}", [], client, 0.6
        )
        assert len(prompts) == 1

    async def test_all_blocks_fit_single_prompt(self):
        client = MockLLMClient(context_window=10_000)
        blocks = [("block1", ["key1"]), ("block2", ["key2"])]
        prompts = await _split_prompts("h", "f {commit_keys}", blocks, client, 0.6)
        assert len(prompts) == 1
        assert "key1" in prompts[0]
        assert "key2" in prompts[0]

    async def test_blocks_split_when_budget_exceeded(self):
        client = MockLLMClient(context_window=100)
        # budget = 60 tokens, char/4
        # block = 200 chars = 50 tokens; 2 blocks = 100 > 60
        big_block = "x" * 200
        blocks = [(big_block, ["k1"]), (big_block, ["k2"])]
        prompts = await _split_prompts("h", "f {commit_keys}", blocks, client, 0.6)
        assert len(prompts) == 2

    async def test_commit_keys_in_correct_prompt(self):
        client = MockLLMClient(context_window=100)
        big_block = "x" * 200
        blocks = [(big_block, ["key1"]), (big_block, ["key2"])]
        prompts = await _split_prompts("h", "f {commit_keys}", blocks, client, 0.6)
        assert "key1" in prompts[0]
        assert "key2" not in prompts[0]
        assert "key2" in prompts[1]
        assert "key1" not in prompts[1]

    async def test_header_repeated_in_each_prompt(self):
        client = MockLLMClient(context_window=100)
        big_block = "x" * 200
        blocks = [(big_block, ["k1"]), (big_block, ["k2"])]
        prompts = await _split_prompts(
            "REGRESSION_HEADER", "footer {commit_keys}", blocks, client, 0.6
        )
        for p in prompts:
            assert "REGRESSION_HEADER" in p

    async def test_three_blocks_into_two_prompts(self):
        client = MockLLMClient(context_window=100)
        big_block = "x" * 160
        blocks = [(big_block, ["k1"]), (big_block, ["k2"]), (big_block, ["k3"])]
        prompts = await _split_prompts("h", "f {commit_keys}", blocks, client, 0.6)
        assert len(prompts) >= 2
        all_keys = " ".join(prompts)
        assert "k1" in all_keys
        assert "k2" in all_keys
        assert "k3" in all_keys


# build_flash_prompts / build_frontier_prompts


class TestBuildFlashPrompts:
    async def test_returns_list_of_prompts(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_flash_prompts(regression_context, diffs, client)
        assert isinstance(prompts, list)
        assert len(prompts) >= 1

    async def test_each_prompt_contains_regression_header(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_flash_prompts(regression_context, diffs, client)
        for p in prompts:
            assert "PERFSCALE-1234" in p
            assert "node-density" in p

    async def test_each_prompt_contains_response_schema(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_flash_prompts(regression_context, diffs, client)
        for p in prompts:
            assert "decisions" in p

    async def test_skips_commits_not_passing_heuristics(self, regression_context):
        regression_context.commits_by_key[COMMIT_KEY].passed_heuristics = False
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_flash_prompts(regression_context, diffs, client)
        for p in prompts:
            assert COMMIT_KEY not in p


class TestBuildFrontierPrompts:
    async def test_returns_list_of_prompts(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_frontier_prompts(
            regression_context, {COMMIT_KEY}, diffs, client
        )
        assert isinstance(prompts, list)
        assert len(prompts) >= 1

    async def test_only_includes_specified_commit_keys(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_frontier_prompts(
            regression_context, {COMMIT_KEY}, diffs, client
        )
        combined = " ".join(prompts)
        assert COMMIT_KEY in combined

    async def test_excludes_unspecified_commit_keys(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_frontier_prompts(regression_context, set(), diffs, client)
        combined = " ".join(prompts)
        assert COMMIT_KEY not in combined

    async def test_prompts_contain_frontier_response_schema(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_frontier_prompts(
            regression_context, {COMMIT_KEY}, diffs, client
        )
        for p in prompts:
            assert "rankings" in p

    async def test_prompts_include_regression_context(self, regression_context):
        diffs = {COMMIT_KEY: {FILE_PATH: "@@ -1 +1 @@ f\n-x\n+y\n"}}
        client = MockLLMClient()
        prompts = await build_frontier_prompts(
            regression_context, {COMMIT_KEY}, diffs, client
        )
        for p in prompts:
            assert "PERFSCALE-1234" in p
