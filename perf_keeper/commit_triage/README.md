# FirstPass Phase 2: Commit Triage

Ranks commits in an OCP payload window by likelihood of causing a performance regression. Regression data (good/bad payload, metric, test, PR list) comes from a Markdown attachment on the JIRA ticket.

## Pipeline

<!-- diagram goes here -->

Heuristic signals: temporal recency (S1), modification intensity (S2), nesting depth shift (S3), control flow delta (S4), change concentration (S5), embedding similarity to the metric description (S6, requires `MetricContext`).

## Files

```
commit_triage/
├── pipeline.py           # entry point: run(context, flash_client, frontier_client)
├── models.py             # RegressionContext, CommitModel, PRModel, FileModel, LLM output types
├── data_acquisition.py   # async GitHub client; build_regression_context() assembles the context
├── heuristics.py         # S1-S6 signal functions and file discard logic
├── llm.py                # GoogleLLMClient (Gemini); create_client() for instantiation
├── prompts.py            # flash/frontier prompt builders; handles context-window splitting
├── report.py             # Markdown report for JIRA attachment and Qdrant/BM25 retrieval
├── config.py             # signal weights, thresholds, model names
├── templates/
│   ├── flash.md          # flash prompt template
│   └── frontier.md       # frontier prompt template
└── tests/                # 274 unit tests
```

## Usage

```python
from perf_keeper.commit_triage.data_acquisition import build_regression_context, make_affected_metric
from perf_keeper.commit_triage.llm import create_client, GEMINI_FLASH, GEMINI_PRO
from perf_keeper.commit_triage import pipeline

context = await build_regression_context(
    github_token=...,
    jira_key="PERFSCALE-5204",
    test="node-density",
    good_version="5.0.0-0.nightly-2026-07-01-125918",
    bad_version="5.0.0-0.nightly-2026-07-01-190408",
    pr_urls=[...],
    primary_metric=make_affected_metric("ovnCPU-northd_avg", change=73.59),
    perf_keeper_report_path=Path("..."),
    workspace_dir=Path("/tmp/perf-keeper/PERFSCALE-5204"),
    jira_labels=[],
)

context = await pipeline.run(
    context,
    flash_client=create_client(GEMINI_FLASH, api_key=...),
    frontier_client=create_client(GEMINI_PRO, api_key=...),
)
```

After the run, `context.commits_by_key` has `triage_score` (0-100) and `confidence` set on each ranked commit. A report is written to `workspace_dir/commit-triage-{jira_key}.md`.

## Tests

```bash
uv run pytest perf_keeper/commit_triage/tests/ -q
```
