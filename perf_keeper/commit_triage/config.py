"""Constants and compiled patterns shared across commit triage modules."""

from __future__ import annotations

import re

GITHUB_API = "https://api.github.com"
PER_PAGE = 100
MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 20
HTTP_TIMEOUT: float = 30.0
PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".go",
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".hxx",
        ".java",
        ".rs",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".rb",
        ".sh",
        ".bash",
        ".zsh",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".zig",
        ".lua",
        ".pl",
        ".pm",
        ".r",
        ".m",
        ".f90",
        ".f",
        ".v",
        ".sv",
    }
)
KEEP_NAMES: frozenset[str] = frozenset({"Dockerfile", "Makefile", "Containerfile"})
KEEP_EXTENSIONS: frozenset[str] = frozenset({".dockerfile"})
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".bmp",
        ".webp",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".lib",
        ".o",
        ".wasm",
        ".pyc",
        ".class",
        ".mp4",
        ".mp3",
        ".wav",
        ".avi",
        ".mov",
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".parquet",
        ".arrow",
    }
)
DOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".adoc",
        ".asciidoc",
        ".tex",
    }
)
CONFIG_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".properties",
        ".xml",
        ".proto",
    }
)

TEST_PATH_RE = re.compile(
    r"(^|/)tests?/"
    r"|_test\."
    r"|test_[^/]"
    r"|\.test\."
    r"|\.spec\."
    r"|_spec\."
    r"|(^|/)__tests__/"
    r"|(^|/)testdata/"
)
GENERATED_PATH_RE = re.compile(
    r"(^|/)vendor/"
    r"|(^|/)generated/"
    r"|_gen\."
    r"|\.pb\.go$"
    r"|(^|/)node_modules/"
    r"|_generated\."
    r"|\.gen\."
    r"|zz_generated"
)

HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?:\s+(.*))?$")
WORD_RE = re.compile(r"\b\w+\b")

EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_OUTPUT_TOKENS: int = 65536
FALLBACK_CONTEXT_WINDOW: int = 1_000_000
CONTEXT_BUDGET_FRACTION: float = 0.6

SIGNAL_WEIGHTS: tuple[float, ...] = (2.0, 1.5, 1.0, 1.5, 1.0)
S6_WEIGHT: float = 0.3
SCORE_THRESHOLD: int = 20
MIN_PASSING_COMMITS: int = 10
MAX_FRONTIER_RANKINGS: int = 15
NEEDS_DEEP_ANALYSIS_TOP_N: int = 5
FRONTIER_MAX: int = 30
DIFF_MIN_THRESHOLD: int = 2

_CF_PYTHON: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "if",
        "elif",
        "else",
        "try",
        "except",
        "finally",
        "with",
        "async",
        "await",
        "yield",
        "return",
        "raise",
        "break",
        "continue",
        "match",
        "case",
    }
)
_CF_GO: frozenset[str] = frozenset(
    {
        "for",
        "if",
        "else",
        "switch",
        "case",
        "select",
        "defer",
        "go",
        "return",
        "break",
        "continue",
        "goto",
        "fallthrough",
    }
)
_CF_RUST: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "loop",
        "if",
        "else",
        "match",
        "return",
        "break",
        "continue",
        "async",
        "await",
    }
)
_CF_JS_TS: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "do",
        "if",
        "else",
        "switch",
        "case",
        "try",
        "catch",
        "finally",
        "throw",
        "return",
        "break",
        "continue",
        "async",
        "await",
        "yield",
    }
)
_CF_JAVA_KT: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "do",
        "if",
        "else",
        "switch",
        "case",
        "try",
        "catch",
        "finally",
        "throw",
        "return",
        "break",
        "continue",
    }
)
_CF_C: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "do",
        "if",
        "else",
        "switch",
        "case",
        "return",
        "break",
        "continue",
        "goto",
    }
)
_CF_CPP: frozenset[str] = _CF_C | frozenset(
    {
        "try",
        "catch",
        "throw",
        "co_await",
        "co_yield",
        "co_return",
    }
)
_CF_RUBY: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "until",
        "if",
        "elsif",
        "else",
        "unless",
        "case",
        "when",
        "begin",
        "rescue",
        "ensure",
        "raise",
        "return",
        "break",
        "next",
        "yield",
    }
)
_CF_SHELL: frozenset[str] = frozenset(
    {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "return",
        "break",
        "continue",
    }
)
CF_DEFAULT: frozenset[str] = frozenset(
    {
        "for",
        "while",
        "if",
        "else",
        "switch",
        "case",
        "try",
        "catch",
        "async",
        "await",
        "yield",
        "return",
        "break",
        "continue",
        "goto",
    }
)
CF_BY_SUFFIX: dict[str, frozenset[str]] = {
    ".py": _CF_PYTHON,
    ".go": _CF_GO,
    ".rs": _CF_RUST,
    ".js": _CF_JS_TS,
    ".jsx": _CF_JS_TS,
    ".ts": _CF_JS_TS,
    ".tsx": _CF_JS_TS,
    ".java": _CF_JAVA_KT,
    ".kt": _CF_JAVA_KT,
    ".scala": _CF_JAVA_KT,
    ".c": _CF_C,
    ".h": _CF_C,
    ".cpp": _CF_CPP,
    ".cc": _CF_CPP,
    ".cxx": _CF_CPP,
    ".hpp": _CF_CPP,
    ".hxx": _CF_CPP,
    ".rb": _CF_RUBY,
    ".sh": _CF_SHELL,
    ".bash": _CF_SHELL,
    ".zsh": _CF_SHELL,
}
