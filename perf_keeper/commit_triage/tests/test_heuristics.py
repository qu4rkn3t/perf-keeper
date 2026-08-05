"""Tests for heuristic filtering and signal computation."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from perf_keeper.commit_triage.heuristics import (
    _aggregate_functions,
    _cosine_similarity,
    _indent_level,
    _mean_indent,
    _parse_hunks,
    s1_temporal_proximity,
    s2_modification_intensity,
    s3_nesting_depth_shift,
    s4_control_flow_delta,
    s5_change_concentration,
    s6_component_proximity,
    should_discard,
)
from perf_keeper.commit_triage.models import (
    CommitModel,
    DiscardReason,
    FileModel,
    MetricContext,
    PRModel,
)
from perf_keeper.commit_triage.models import AffectedMetric, RegressionContext
from perf_keeper.commit_triage.prompts import load_all_diffs


def _minimal_context(tmp_path, commits_by_key, files_by_key, primary_metric=None):
    pm = primary_metric or AffectedMetric(name="cpu", change=-5.0)
    return RegressionContext(
        analysis_id="t",
        workspace_dir=tmp_path,
        jira_key="P-1",
        test="t",
        primary_metric=pm,
        good_version="4.17",
        bad_version="4.18",
        perf_keeper_report_path=tmp_path / "r.md",
        commits_by_key=commits_by_key,
        files_by_key=files_by_key,
    )


T_MIN = datetime(2025, 1, 1, tzinfo=timezone.utc)
T_MID = datetime(2025, 1, 15, tzinfo=timezone.utc)
T_MAX = datetime(2025, 1, 30, tzinfo=timezone.utc)


def _file(
    path: str, tmp_path: Path, discarded: bool = False, diff_text: str = ""
) -> FileModel:
    diff_p = tmp_path / "diffs" / "sha12345" / "f.diff"
    diff_p.parent.mkdir(parents=True, exist_ok=True)
    if diff_text:
        diff_p.write_text(diff_text, encoding="utf-8")
    return FileModel(
        file_key=f"pr:sha:{path}",
        commit_key="pr:sha",
        path=path,
        diff_path=diff_p,
        additions=5,
        deletions=2,
        discarded=discarded,
    )


def _commit(file_keys: list[str], committed_at=None, passed=True) -> CommitModel:
    return CommitModel(
        commit_key="pr:sha",
        sha="sha",
        message="fix",
        pr_key="pr",
        committed_at=committed_at or T_MID,
        file_keys=file_keys,
        passed_heuristics=passed,
    )


# should_discard


class TestShouldDiscard:
    @pytest.mark.parametrize(
        "path",
        [
            "main.go",
            "pkg/server/server.py",
            "src/lib.rs",
            "Controller.java",
            "index.ts",
            "component.tsx",
            "helper.js",
            "util.cpp",
            "driver.c",
            "lib.h",
            "app.rb",
            "script.sh",
            "module.cs",
            "lib.swift",
        ],
    )
    def test_keeps_code_files(self, path, tmp_path):
        assert should_discard(_file(path, tmp_path)) is None

    @pytest.mark.parametrize("name", ["Dockerfile", "Makefile", "Containerfile"])
    def test_keeps_named_files(self, name, tmp_path):
        assert should_discard(_file(name, tmp_path)) is None

    @pytest.mark.parametrize(
        "path",
        [
            "image.png",
            "photo.jpg",
            "icon.gif",
            "lib.so",
            "binary.exe",
            "archive.tar.gz",
            "font.woff2",
            "data.parquet",
            "compiled.pyc",
            "bytecode.class",
        ],
    )
    def test_discards_binary(self, path, tmp_path):
        assert should_discard(_file(path, tmp_path)) == DiscardReason.BINARY

    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "docs/guide.rst",
            "CHANGELOG.txt",
            "notes.adoc",
            "paper.tex",
        ],
    )
    def test_discards_docs(self, path, tmp_path):
        assert should_discard(_file(path, tmp_path)) == DiscardReason.DOCS

    @pytest.mark.parametrize(
        "path",
        [
            "config.yaml",
            "settings.json",
            "setup.toml",
            "app.xml",
            ".env",
            "schema.proto",
            "app.ini",
        ],
    )
    def test_discards_config(self, path, tmp_path):
        assert should_discard(_file(path, tmp_path)) == DiscardReason.CONFIG

    @pytest.mark.parametrize(
        "path",
        [
            "vendor/github.com/foo/bar.go",
            "generated/types.go",
            "api.pb.go",
            "node_modules/lodash/index.js",
            "zz_generated.deepcopy.go",
            "types_gen.go",
            "auto_generated.py",
        ],
    )
    def test_discards_generated(self, path, tmp_path):
        assert should_discard(_file(path, tmp_path)) == DiscardReason.CONFIG

    @pytest.mark.parametrize(
        "path",
        [
            "pkg/server/server_test.go",
            "tests/test_server.py",
            "src/__tests__/component.spec.ts",
            "utils/helper.test.js",
            "testdata/fixture.go",
        ],
    )
    def test_discards_test_files(self, path, tmp_path):
        assert should_discard(_file(path, tmp_path)) == DiscardReason.TEST_FILE

    def test_code_in_vendor_discarded(self, tmp_path):
        assert should_discard(_file("vendor/main.go", tmp_path)) == DiscardReason.CONFIG

    def test_unknown_extension_discarded(self, tmp_path):
        assert (
            should_discard(_file("file.unknownext", tmp_path)) == DiscardReason.CONFIG
        )


# load_diffs


class TestLoadAllDiffs:
    def test_loads_non_discarded_files(self, tmp_path):
        diff_text = "@@ -1 +1 @@ func\n-old\n+new\n"
        commit_key, path = "pr:sha", "main.go"
        file_key = f"{commit_key}:{path}"
        diff_p = tmp_path / "diffs" / "sha12345" / "f.diff"
        diff_p.parent.mkdir(parents=True, exist_ok=True)
        diff_p.write_text(diff_text)
        f = FileModel(
            file_key=file_key, commit_key=commit_key, path=path, diff_path=diff_p
        )
        c = CommitModel(
            commit_key=commit_key,
            sha="sha",
            message="m",
            pr_key="pr",
            file_keys=[file_key],
        )
        ctx = _minimal_context(tmp_path, {commit_key: c}, {file_key: f})
        result = load_all_diffs(ctx)
        assert commit_key in result
        assert path in result[commit_key]
        assert result[commit_key][path] == diff_text

    def test_excludes_discarded_files(self, tmp_path):
        commit_key, path = "pr:sha", "main.go"
        file_key = f"{commit_key}:{path}"
        diff_p = tmp_path / "f.diff"
        diff_p.write_text("diff")
        f = FileModel(
            file_key=file_key,
            commit_key=commit_key,
            path=path,
            diff_path=diff_p,
            discarded=True,
        )
        c = CommitModel(
            commit_key=commit_key,
            sha="sha",
            message="m",
            pr_key="pr",
            file_keys=[file_key],
        )
        ctx = _minimal_context(tmp_path, {commit_key: c}, {file_key: f})
        assert load_all_diffs(ctx) == {}

    def test_excludes_missing_diff_files(self, tmp_path):
        commit_key, path = "pr:sha", "main.go"
        file_key = f"{commit_key}:{path}"
        f = FileModel(
            file_key=file_key,
            commit_key=commit_key,
            path=path,
            diff_path=tmp_path / "missing.diff",
        )
        c = CommitModel(
            commit_key=commit_key,
            sha="sha",
            message="m",
            pr_key="pr",
            file_keys=[file_key],
        )
        ctx = _minimal_context(tmp_path, {commit_key: c}, {file_key: f})
        assert load_all_diffs(ctx) == {}


# diff parsing


class TestParseHunks:
    def test_empty_diff_returns_empty(self):
        assert _parse_hunks("f.go", "") == []

    def test_no_hunk_headers(self):
        assert _parse_hunks("f.go", "+added\n-removed\n") == []

    def test_single_hunk_with_function(self):
        diff = "@@ -1,2 +1,3 @@ func process() {\n-old\n+new\n+extra\n"
        hunks = _parse_hunks("f.go", diff)
        assert len(hunks) == 1
        assert hunks[0].function == "func process() {"
        assert hunks[0].removed == ["old"]
        assert hunks[0].added == ["new", "extra"]
        assert hunks[0].file_path == "f.go"

    def test_hunk_without_function(self):
        hunks = _parse_hunks("f.py", "@@ -1,1 +1,2 @@\n-old\n+new\n")
        assert len(hunks) == 1
        assert hunks[0].function == ""

    def test_multiple_hunks(self):
        diff = "@@ -1,1 +1,1 @@ func_a\n-a\n+b\n@@ -10,1 +10,1 @@ func_b\n-x\n+y\n"
        hunks = _parse_hunks("f.go", diff)
        assert len(hunks) == 2
        assert hunks[0].function == "func_a"
        assert hunks[1].function == "func_b"

    def test_context_lines_not_collected(self):
        diff = "@@ -1,3 +1,3 @@ myfunc\n context\n-old\n+new\n context2\n"
        hunks = _parse_hunks("f.go", diff)
        assert hunks[0].added == ["new"]
        assert hunks[0].removed == ["old"]


class TestAggregateFunctions:
    def test_merges_same_function_same_file(self):
        diff = (
            "@@ -1,1 +1,2 @@ myfunc\n-x\n+y\n+z\n@@ -10,1 +11,2 @@ myfunc\n-a\n+b\n+c\n"
        )
        funcs = _aggregate_functions({"file.go": diff})
        assert len(funcs) == 1
        assert len(funcs[0].added) == 4
        assert len(funcs[0].removed) == 2

    def test_different_functions_separate(self):
        diff = "@@ -1,1 +1,1 @@ func_a\n-x\n+y\n@@ -10,1 +10,1 @@ func_b\n-a\n+b\n"
        funcs = _aggregate_functions({"file.go": diff})
        assert len(funcs) == 2

    def test_same_name_different_files_separate(self):
        diff = "@@ -1,1 +1,1 @@ process\n-x\n+y\n"
        funcs = _aggregate_functions({"a.go": diff, "b.go": diff})
        assert len(funcs) == 2
        paths = {f.file_path for f in funcs}
        assert "a.go" in paths and "b.go" in paths

    def test_empty_diffs_empty_result(self):
        assert _aggregate_functions({}) == []


# indentation helpers


class TestIndentHelpers:
    def test_no_indent(self):
        assert _indent_level("x = 1") == 0.0

    def test_space_indent(self):
        assert _indent_level("    x = 1") == 4.0

    def test_tab_normalized_to_four(self):
        assert _indent_level("\tx = 1") == 4.0

    def test_mixed_indent(self):
        assert _indent_level("\t    x") == 8.0

    def test_mean_indent_empty(self):
        assert _mean_indent([]) == 0.0

    def test_mean_indent_uniform(self):
        assert _mean_indent(["    a", "    b", "    c"]) == pytest.approx(4.0)


# S1: temporal proximity


class TestS1TemporalProximity:
    def test_at_t_min(self):
        c = _commit([], committed_at=T_MIN)
        assert s1_temporal_proximity(c, T_MIN, T_MAX) == pytest.approx(0.0)

    def test_at_t_max(self):
        c = _commit([], committed_at=T_MAX)
        assert s1_temporal_proximity(c, T_MIN, T_MAX) == pytest.approx(1.0)

    def test_midpoint_between_zero_and_one(self):
        c = _commit([], committed_at=T_MID)
        score = s1_temporal_proximity(c, T_MIN, T_MAX)
        assert 0.0 < score < 1.0

    def test_equal_timestamps_returns_half(self):
        c = _commit([], committed_at=T_MIN)
        assert s1_temporal_proximity(c, T_MIN, T_MIN) == pytest.approx(0.5)

    def test_none_committed_at_returns_half(self):
        c = CommitModel(commit_key="p:s", sha="s", message="m", pr_key="p")
        assert s1_temporal_proximity(c, T_MIN, T_MAX) == pytest.approx(0.5)

    def test_proportional_to_time(self):
        quarter = datetime(2025, 1, 8, tzinfo=timezone.utc)
        c = _commit([], committed_at=quarter)
        score = s1_temporal_proximity(c, T_MIN, T_MAX)
        assert score == pytest.approx(7 / 29, abs=0.01)


# S2: modification intensity


class TestS2ModificationIntensity:
    def test_empty_diffs(self):
        assert s2_modification_intensity({}) == pytest.approx(0.0)

    def test_no_hunk_headers(self):
        assert s2_modification_intensity({"f.go": "no hunks"}) == pytest.approx(0.0)

    def test_pure_addition(self):
        diff = "@@ -1,1 +1,2 @@ f\n context\n+new line\n"
        assert s2_modification_intensity({"f.go": diff}) == pytest.approx(0.0)

    def test_pure_deletion(self):
        diff = "@@ -1,2 +1,1 @@ f\n context\n-removed\n"
        assert s2_modification_intensity({"f.go": diff}) == pytest.approx(0.0)

    def test_balanced_rewrite_is_one(self):
        diff = "@@ -1,2 +1,2 @@ f\n-old line\n+new line\n"
        assert s2_modification_intensity({"f.go": diff}) == pytest.approx(1.0)

    def test_two_added_one_removed(self):
        diff = "@@ -1,3 +1,4 @@ f\n-x\n+a\n+b\n"
        score = s2_modification_intensity({"f.go": diff})
        assert score == pytest.approx(2 * min(2, 1) / (2 + 1))

    def test_multiple_functions_averaged(self):
        diff = (
            "@@ -1,2 +1,2 @@ func_a\n-a\n+b\n"
            "@@ -10,1 +10,2 @@ func_b\n context\n+extra\n"
        )
        score = s2_modification_intensity({"f.go": diff})
        assert 0.0 < score < 1.0


# S3: nesting depth shift


class TestS3NestingDepthShift:
    def test_empty_diffs(self):
        assert s3_nesting_depth_shift({}) == pytest.approx(0.0)

    def test_no_hunks(self):
        assert s3_nesting_depth_shift({"f.py": "plain text"}) == pytest.approx(0.0)

    def test_added_lines_deeper(self):
        diff = "@@ -1,2 +1,2 @@ f\n-x = 1\n+    x = 1\n"
        assert s3_nesting_depth_shift({"f.py": diff}) > 0.0

    def test_added_lines_shallower_clamped_to_zero(self):
        diff = "@@ -1,2 +1,2 @@ f\n-    x = 1\n+x = 1\n"
        assert s3_nesting_depth_shift({"f.py": diff}) == pytest.approx(0.0)

    def test_equal_indentation_near_zero(self):
        diff = "@@ -1,2 +1,2 @@ f\n-    x = 1\n+    x = 2\n"
        assert s3_nesting_depth_shift({"f.py": diff}) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        diff = "@@ -1,2 +1,2 @@ f\n-x\n+                x\n"
        score = s3_nesting_depth_shift({"f.py": diff})
        assert 0.0 <= score <= 1.0

    def test_tanh_applied(self):
        diff = "@@ -1,2 +1,2 @@ f\n-x = 1\n+    x = 1\n"
        score = s3_nesting_depth_shift({"f.py": diff})
        expected = max(0.0, min(1.0, math.tanh(4.0)))
        assert score == pytest.approx(expected, abs=0.001)


# S4: control flow delta


class TestS4ControlFlowDelta:
    def test_empty_diffs(self):
        assert s4_control_flow_delta({}) == pytest.approx(0.0)

    def test_no_control_flow_changes(self):
        diff = "@@ -1,2 +1,2 @@ f\n-x = old\n+x = new\n"
        assert s4_control_flow_delta({"f.py": diff}) == pytest.approx(0.0)

    def test_ten_new_if_statements_saturates(self):
        lines = "".join(f"+    if cond{i}: pass\n" for i in range(10))
        diff = f"@@ -1,1 +1,11 @@ func\n-x = 1\n{lines}"
        assert s4_control_flow_delta({"f.py": diff}) == pytest.approx(1.0)

    def test_five_new_cf_lines_gives_half(self):
        lines = "".join(f"+    if cond{i}: pass\n" for i in range(5))
        diff = f"@@ -1,1 +1,6 @@ func\n-x = 1\n{lines}"
        score = s4_control_flow_delta({"f.py": diff})
        assert score == pytest.approx(0.5, abs=0.05)

    def test_more_removed_than_added_clamped_to_zero(self):
        diff = "@@ -1,4 +1,1 @@ f\n-    if a: pass\n-    if b: pass\n-    if c: pass\n+x = 1\n"
        assert s4_control_flow_delta({"f.py": diff}) == pytest.approx(0.0)

    def test_go_keywords_recognized(self):
        diff = "@@ -1,1 +1,3 @@ func\n-x := 1\n+if err != nil {\n+\treturn err\n+}\n"
        assert s4_control_flow_delta({"f.go": diff}) > 0.0

    def test_result_clamped_to_unit_interval(self):
        lines = "".join(f"+    if cond{i}: pass\n" for i in range(50))
        diff = f"@@ -1,1 +1,51 @@ func\n-x\n{lines}"
        score = s4_control_flow_delta({"f.py": diff})
        assert 0.0 <= score <= 1.0

    def test_language_dispatch_rust(self):
        diff = "@@ -1,1 +1,2 @@ fn process\n-x\n+    match value { _ => {} }\n"
        assert s4_control_flow_delta({"lib.rs": diff}) > 0.0


# S5: change concentration


class TestS5ChangeConcentration:
    def _file_key(self, path):
        return f"pr:sha:{path}"

    def _make_file(self, path, tmp_path, additions=0, deletions=0, discarded=False):
        fk = self._file_key(path)
        return fk, FileModel(
            file_key=fk,
            commit_key="pr:sha",
            path=path,
            diff_path=tmp_path / "x.diff",
            additions=additions,
            deletions=deletions,
            discarded=discarded,
        )

    def test_no_files(self):
        c = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[]
        )
        assert s5_change_concentration(c, {}) == pytest.approx(0.0)

    def test_zero_changes(self, tmp_path):
        fk, f = self._make_file("f.go", tmp_path)
        c = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[fk]
        )
        assert s5_change_concentration(c, {fk: f}) == pytest.approx(0.0)

    def test_known_value(self, tmp_path):
        fk, f = self._make_file("f.go", tmp_path, additions=25, deletions=25)
        c = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[fk]
        )
        score = s5_change_concentration(c, {fk: f})
        assert score == pytest.approx(math.tanh(50 / 50))

    def test_high_concentration_near_one(self, tmp_path):
        fk, f = self._make_file("f.go", tmp_path, additions=500, deletions=500)
        c = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[fk]
        )
        assert s5_change_concentration(c, {fk: f}) > 0.99

    def test_discarded_files_excluded(self, tmp_path):
        # File B has 0 changes. When it is active, n=2 which lowers concentration.
        # When discarded, n=1 which raises concentration (tanh(50/50) > tanh(50/100)).
        fk1, f1 = self._make_file("a.go", tmp_path, additions=25, deletions=25)
        fk_disc, f_disc = self._make_file(
            "b.go", tmp_path, additions=0, deletions=0, discarded=True
        )
        fk_act, f_act = self._make_file(
            "c.go", tmp_path, additions=0, deletions=0, discarded=False
        )
        c_disc = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[fk1, fk_disc]
        )
        c_act = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[fk1, fk_act]
        )
        score_discarded = s5_change_concentration(c_disc, {fk1: f1, fk_disc: f_disc})
        score_active = s5_change_concentration(c_act, {fk1: f1, fk_act: f_act})
        assert score_discarded > score_active

    def test_spread_across_many_files_lower_score(self, tmp_path):
        single_key, single_f = self._make_file(
            "a.go", tmp_path, additions=100, deletions=100
        )
        c_single = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=[single_key]
        )
        score_single = s5_change_concentration(c_single, {single_key: single_f})

        keys_and_files = [
            self._make_file(f"f{i}.go", tmp_path, additions=10, deletions=10)
            for i in range(10)
        ]
        all_keys = [k for k, _ in keys_and_files]
        all_files = {k: f for k, f in keys_and_files}
        c_spread = CommitModel(
            commit_key="p:s", sha="s", message="m", pr_key="p", file_keys=all_keys
        )
        score_spread = s5_change_concentration(c_spread, all_files)
        assert score_single > score_spread


# cosine similarity helper


class TestCosineSimilarity:
    def test_identical_unit_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_clamped_to_zero(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_partial_similarity(self):
        import math

        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=0.001)


# S6: component proximity (embedding-based)


class TestS6ComponentProximity:
    def _commit_and_pr(self, message="fix", title="PR"):
        c = CommitModel(commit_key="p:s", sha="s", message=message, pr_key="p")
        pr = PRModel(pr_key="p", owner="o", repo="r", number=1, title=title)
        return c, pr

    async def test_no_context_is_not_called(self):
        """S6 should not be called by the pipeline when MetricContext is absent."""
        # Callers are responsible for not invoking s6 without a MetricContext.
        # This test documents that the pipeline guards the call site, not s6 itself.
        pass

    async def test_returns_float_in_unit_interval(self, mock_client):
        ctx = MetricContext(
            name="kubelet",
            description="kubelet performance",
            known_causes=["scheduler"],
        )
        c, pr = self._commit_and_pr(message="fix kubelet scheduling")
        score = await s6_component_proximity(c, pr, {}, {}, ctx, mock_client.embed)
        assert 0.0 <= score <= 1.0

    async def test_identical_texts_give_one(self):
        ctx = MetricContext(
            name="etcd", description="etcd latency fsync", known_causes=["disk io"]
        )
        c, pr = self._commit_and_pr(message="etcd latency fsync disk io")

        async def embed(text):
            return [1.0, 0.0]

        score = await s6_component_proximity(c, pr, {}, {}, ctx, embed)
        assert score == pytest.approx(1.0)

    async def test_orthogonal_embeddings_give_zero(self):
        ctx = MetricContext(name="etcd", description="etcd latency", known_causes=[])

        async def embed(text):
            return [1.0, 0.0] if "etcd" in text else [0.0, 1.0]

        c, pr = self._commit_and_pr(message="unrelated change", title="whitespace")
        score = await s6_component_proximity(c, pr, {}, {}, ctx, embed)
        assert score == pytest.approx(0.0)

    async def test_embed_called_twice_concurrently(self):
        ctx = MetricContext(name="ovn", description="ovn cpu usage", known_causes=[])
        c, pr = self._commit_and_pr(message="ovn controller fix")
        call_log: list[str] = []

        async def embed(text):
            call_log.append(text)
            return [0.5, 0.5]

        await s6_component_proximity(c, pr, {}, {}, ctx, embed)
        assert len(call_log) == 2

    async def test_function_names_included_in_commit_text(self):
        ctx = MetricContext(
            name="syncPod", description="syncPod latency", known_causes=[]
        )
        c, pr = self._commit_and_pr(message="fix scheduling")
        diffs = {"main.go": "@@ -1 +1 @@ func syncPod() error {\n-x\n+y\n"}
        received: list[str] = []

        async def embed(text):
            received.append(text)
            return [0.7, 0.3]

        await s6_component_proximity(c, pr, {}, diffs, ctx, embed)
        assert any("syncPod" in t for t in received)
