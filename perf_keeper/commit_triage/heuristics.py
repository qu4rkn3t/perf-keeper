"""Heuristic filtering and signal computation for commit triage."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from perf_keeper.commit_triage.config import (
    BINARY_EXTENSIONS,
    CF_BY_SUFFIX,
    CF_DEFAULT,
    CODE_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DOC_EXTENSIONS,
    GENERATED_PATH_RE,
    HUNK_HEADER_RE,
    KEEP_EXTENSIONS,
    KEEP_NAMES,
    TEST_PATH_RE,
    WORD_RE,
)
from perf_keeper.commit_triage.models import (
    CommitModel,
    DiscardReason,
    FileModel,
    MetricContext,
    PRModel,
)

logger = logging.getLogger(__name__)

# Signal functions s1-s6 are intentionally unlogged; they are called per-commit
# in the hot path and logging inside them would generate excessive noise.

EmbedFn = Callable[[str], Awaitable[list[float]]]


def should_discard(file: FileModel) -> DiscardReason | None:
    """Return a DiscardReason to exclude this file, or None to keep it."""
    p = Path(file.path)
    suffix = p.suffix.lower()
    name = p.name
    path = file.path

    if suffix in BINARY_EXTENSIONS:
        logger.debug("Discarded %s: %s", file.path, DiscardReason.BINARY.value)
        return DiscardReason.BINARY

    if GENERATED_PATH_RE.search(path):
        logger.debug("Discarded %s: %s", file.path, DiscardReason.CONFIG.value)
        return DiscardReason.CONFIG

    if name in KEEP_NAMES or suffix in KEEP_EXTENSIONS:
        return None

    if any(name == k or name.startswith(k + ".") for k in KEEP_NAMES):
        return None

    if suffix in CODE_EXTENSIONS:
        if TEST_PATH_RE.search(path):
            logger.debug("Discarded %s: %s", file.path, DiscardReason.TEST_FILE.value)
            return DiscardReason.TEST_FILE
        return None

    if suffix in DOC_EXTENSIONS:
        logger.debug("Discarded %s: %s", file.path, DiscardReason.DOCS.value)
        return DiscardReason.DOCS

    if suffix in CONFIG_EXTENSIONS:
        logger.debug("Discarded %s: %s", file.path, DiscardReason.CONFIG.value)
        return DiscardReason.CONFIG

    logger.debug("Discarded %s: %s", file.path, DiscardReason.CONFIG.value)
    return DiscardReason.CONFIG


@dataclass
class _Hunk:
    function: str
    file_path: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def _parse_hunks(file_path: str, diff_text: str) -> list[_Hunk]:
    """Parse a unified diff into hunks labelled with their enclosing function."""
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    for line in diff_text.splitlines():
        m = HUNK_HEADER_RE.match(line)
        if m:
            current = _Hunk(function=m.group(1) or "", file_path=file_path)
            hunks.append(current)
        elif current is not None:
            if line.startswith("+"):
                current.added.append(line[1:])
            elif line.startswith("-"):
                current.removed.append(line[1:])
    return hunks


def _aggregate_functions(diffs: dict[str, str]) -> list[_Hunk]:
    """Merge hunks across all diffs, grouping by (file_path, function_name)."""
    merged: dict[str, _Hunk] = {}
    for file_path, diff_text in diffs.items():
        for hunk in _parse_hunks(file_path, diff_text):
            key = f"{file_path}::{hunk.function}"
            if key not in merged:
                merged[key] = _Hunk(function=hunk.function, file_path=file_path)
            merged[key].added.extend(hunk.added)
            merged[key].removed.extend(hunk.removed)
    return list(merged.values())


def _indent_level(line: str) -> float:
    count = 0.0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4
        else:
            break
    return count


def _mean_indent(lines: list[str]) -> float:
    return sum(_indent_level(line) for line in lines) / len(lines) if lines else 0.0


def _control_flow_keywords(file_path: str) -> frozenset[str]:
    return CF_BY_SUFFIX.get(Path(file_path).suffix.lower(), CF_DEFAULT)


def _cf_line_count(lines: list[str], keywords: frozenset[str]) -> int:
    """Count lines containing at least one control flow keyword."""
    return sum(1 for line in lines if set(WORD_RE.findall(line)) & keywords)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return max(0.0, dot / (mag_a * mag_b)) if mag_a > 0 and mag_b > 0 else 0.0


def s1_temporal_proximity(
    commit: CommitModel, t_min: datetime, t_max: datetime
) -> float:
    """S1 ∈ [0, 1]: Score newer commits higher; falls back to 0.5 when range is zero."""
    if commit.committed_at is None or t_min == t_max:
        return 0.5
    span = (t_max - t_min).total_seconds()
    return (commit.committed_at - t_min).total_seconds() / span


def s2_modification_intensity(diffs: dict[str, str]) -> float:
    """S2 ∈ [0, 1]: Score rewrite-style changes (balanced adds/deletes) higher."""
    functions = _aggregate_functions(diffs)
    if not functions:
        return 0.0
    total = 0.0
    count = 0
    for h in functions:
        a, d = len(h.added), len(h.removed)
        if a + d == 0:
            continue
        total += 2 * min(a, d) / (a + d)
        count += 1
    return total / count if count else 0.0


def s3_nesting_depth_shift(diffs: dict[str, str]) -> float:
    """S3 ∈ [0, 1]: Score commits that add deeper nesting to modified functions."""
    functions = _aggregate_functions(diffs)
    if not functions:
        return 0.0
    total_shift = sum(
        _mean_indent(h.added) - _mean_indent(h.removed) for h in functions
    )
    return max(0.0, min(1.0, math.tanh(total_shift / len(functions))))


def s4_control_flow_delta(diffs: dict[str, str]) -> float:
    """S4 ∈ [0, 1]: Net new control flow statements per function, language-aware."""
    functions = _aggregate_functions(diffs)
    if not functions:
        return 0.0
    total = sum(
        _cf_line_count(h.added, _control_flow_keywords(h.file_path))
        - _cf_line_count(h.removed, _control_flow_keywords(h.file_path))
        for h in functions
    )
    avg = total / len(functions)
    return min(1.0, max(0.0, avg) / 10.0)


def s5_change_concentration(
    commit: CommitModel, files_by_key: dict[str, FileModel]
) -> float:
    """S5 ∈ (0, 1): Score commits with many changes concentrated in few files."""
    active = [
        files_by_key[k]
        for k in commit.file_keys
        if k in files_by_key and not files_by_key[k].discarded
    ]
    n = len(active)
    a = sum(f.additions for f in active)
    d = sum(f.deletions for f in active)
    return math.tanh((a + d) / (max(n, 1) * 50))


async def s6_component_proximity(
    commit: CommitModel,
    pr: PRModel,
    files_by_key: dict[str, FileModel],
    diffs: dict[str, str],
    metric_context: MetricContext,
    embed: EmbedFn,
) -> float:
    """S6 ∈ [0, 1]: Embedding cosine similarity between commit context and the regressing metric."""
    k_text = " ".join(
        [
            metric_context.name,
            metric_context.description,
            *metric_context.known_causes,
        ]
    )

    all_paths = [files_by_key[k].path for k in commit.file_keys if k in files_by_key]
    t_parts: list[str] = [commit.message, pr.title, pr.description, *all_paths]
    for file_path, diff_text in diffs.items():
        t_parts.extend(h.function for h in _parse_hunks(file_path, diff_text))
    t_text = " ".join(t_parts)

    k_emb, t_emb = await asyncio.gather(embed(k_text), embed(t_text))
    return _cosine_similarity(k_emb, t_emb)
