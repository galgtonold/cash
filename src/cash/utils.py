"""Shared utility functions for the cash library.

``normalize_path`` — pure path helper used throughout the codebase.

Notebook-specific I/O and Jupyter Server HTTP discovery
(``get_notebook_path``, ``get_notebook_cells``, etc.) live in
``cash.notebook.server_discovery``.  For backward compatibility these names
are still accessible here, but new code should import them from
``cash.notebook.server_discovery`` directly.
"""

from __future__ import annotations

import logging
import os
import sys

from cash.notebook.server_discovery import (  # noqa: F401  (re-exports)
    extract_notebook_path_from_vscode_cell_id,
    get_notebook_cells,
    get_notebook_cells_with_ids,
    get_notebook_path,
    invalidate_notebook_path_cache,
    set_notebook_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_path",
    "resolve_file_dep_path",
    "safe_text",
    "stdout_supports_unicode",
    "invalidate_notebook_path_cache",
    "set_notebook_path",
    "extract_notebook_path_from_vscode_cell_id",
    "get_notebook_path",
    "get_notebook_cells",
    "get_notebook_cells_with_ids",
]


def normalize_path(path: str) -> str:
    """Return *path* with all OS-native separators replaced by forward slashes.

    Used to produce portable, platform-independent path strings for cache keys
    and dependency tracking.  On POSIX systems this is a no-op; on Windows it
    converts backslashes to forward slashes.

    Examples::

        normalize_path("C:\\\\Users\\\\foo\\\\bar.csv")  # → "C:/Users/foo/bar.csv"
        normalize_path("/home/foo/bar.csv")              # → "/home/foo/bar.csv"
    """
    return path.replace(os.path.sep, '/')


# ---------------------------------------------------------------------------
# Console encoding helpers
# ---------------------------------------------------------------------------
#
# On Windows, the default Python REPL stdout uses cp1252 which cannot encode
# emoji code points like ✅ or ⚙️.  Printing such characters raises
# ``UnicodeEncodeError: 'charmap' codec can't encode character ...``.
#
# Inside Jupyter / IPython kernels stdout is always UTF-8 (the kernel encodes
# bytes for the front-end), so emojis render fine there.  The crash only
# affects users who do ``import cash`` and call our magics from a plain
# Windows ``python.exe`` shell.
#
# ``safe_text`` lets call sites keep the readable, emoji-rich strings while
# silently downgrading them to ASCII fallbacks when the active stdout cannot
# encode them.  ``stdout_supports_unicode`` is also exposed so callers can
# branch up-front (e.g. choose a different code path entirely).

_ASCII_FALLBACKS: dict[str, str] = {
    # Status / outcome
    "✅": "[OK]",
    "❌": "[X]",
    "⚠️": "[!]",
    "⚠": "[!]",
    "✓": "[v]",
    "❓": "[?]",
    "🚫": "[no-cache]",
    # Cache lifecycle
    "⚡": "[cached]",
    "⚙️": "[run]",
    "⏩": "[skip]",
    "⏳": "[wait]",
    "🔄": "[refresh]",
    "♻️": "[reuse]",
    "⬆️": "[upstream]",
    "🔁": "[loop]",
    # Decorations / arrows
    "→": "->",
    "←": "<-",
    "↑": "^",
    "↓": "v",
    "↔": "<->",
    "↻": "~>",
    "└": "L",
    "─": "-",
    "│": "|",
    "├": "+",
    "…": "...",
    "∈": "in",
    # Dashboards / debug
    "🔧": "[computed]",
    "📦": "[restored]",
    "📋": "[provenance]",
    "📊": "[stats]",
    "📈": "[trend-up]",
    "📉": "[trend-down]",
    "📁": "[dir]",
    "🐛": "[bug]",
    "🏷️": "[tag]",
    "⏭️": "[next]",
    "🎯": "[target]",
}


def stdout_supports_unicode(stream: object | None = None) -> bool:
    """Return ``True`` if *stream* (default: ``sys.stdout``) can encode emojis.

    Cheap and side-effect free.  Used by :func:`safe_text` to decide whether
    to pass the input through unchanged or downgrade it to ASCII fallbacks.
    """
    if stream is None:
        stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "ascii"
    encoding_lc = encoding.lower()
    if encoding_lc.startswith("utf"):
        return True
    try:
        # ✅ and ⚙️ together cover both single-codepoint emoji and the
        # variation-selector form most likely to break under cp1252 / latin-1.
        "✅⚙️".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def safe_text(s: str, *, stream: object | None = None) -> str:
    """Return *s* with characters un-encodable by *stream* replaced by ASCII.

    Pass-through when the stream can encode everything (the common case in
    Jupyter / on Linux / when ``PYTHONIOENCODING=utf-8``).  Otherwise replace
    each unsupported character with an entry from :data:`_ASCII_FALLBACKS`
    or, lacking a mapping, drop it.

    The function preserves all ASCII characters as-is, so log lines stay
    readable even on legacy Windows consoles.
    """
    if not s:
        return s
    if stream is None:
        stream = sys.stdout
    if stdout_supports_unicode(stream):
        return s
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        s.encode(encoding)
        return s  # nothing to downgrade
    except UnicodeEncodeError:
        pass
    out: list[str] = []
    for ch in s:
        try:
            ch.encode(encoding)
        except UnicodeEncodeError:
            out.append(_ASCII_FALLBACKS.get(ch, ""))
        else:
            out.append(ch)
    return "".join(out)


def resolve_file_dep_path(stored_path: str) -> str | None:
    """Resolve a stored file dependency path, trying fallbacks if it doesn't exist.

    When a project is moved (e.g. Google Drive path changes), the absolute path
    stored in cache metadata may no longer be valid.  This function tries to
    locate the file at alternative paths:

    1. The stored path as-is.
    2. The basename resolved relative to the current working directory.
    3. Progressively longer path suffixes relative to the current working directory
       (handles subdirectory structure like ``examples/data.csv``).

    Returns the resolved path if found, or ``None`` if the file cannot be located.
    """
    if os.path.exists(stored_path):
        return stored_path

    basename = os.path.basename(stored_path)

    # Fallback 1: basename in CWD
    cwd_candidate = os.path.join(os.getcwd(), basename)
    if os.path.exists(cwd_candidate):
        return normalize_path(os.path.realpath(cwd_candidate))

    # Fallback 2: try progressively longer path suffixes relative to CWD.
    # E.g. stored = "C:/old/root/project/examples/data.csv"
    #   → try "examples/data.csv" relative to CWD
    parts = stored_path.replace('\\', '/').split('/')
    # Start from the second-to-last component (parent dir + filename)
    for i in range(max(len(parts) - 2, 1), 0, -1):
        suffix = '/'.join(parts[i:])
        candidate = os.path.join(os.getcwd(), suffix)
        if os.path.exists(candidate):
            return normalize_path(os.path.realpath(candidate))

    return None
