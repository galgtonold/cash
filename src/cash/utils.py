"""Shared utility functions for the cash library.

Pure helpers that don't depend on Jupyter or filesystem discovery.
Notebook-specific I/O and Jupyter Server HTTP discovery live in
``cash.notebook.server_discovery``.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "is_remote_url",
    "replace_with_retry",
    "normalize_path",
    "resolve_file_dep_path",
    "safe_text",
    "stdout_supports_unicode",
]

# ``file://`` is excluded: it names a local path that can genuinely be stat'ed.
_URL_SCHEME_RE = re.compile(r"^(?!file://)[a-zA-Z][a-zA-Z0-9+.\-]*://")


def is_remote_url(path: str) -> bool:
    """Whether *path* is a remote URL rather than a local filesystem path.

    A single definition, because the answer decides behaviour in a dozen
    places: whether the file tracker records a read on its remote channel,
    whether a dependency entry is validated by stat or by the store's
    validator, and whether a stored key can be path-resolved at all.
    """
    return bool(_URL_SCHEME_RE.match(path))


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


def _basename_candidates(stored_path: str) -> list[str]:
    """Basenames to try for *stored_path*, most-trusted first.

    The host reading comes first so a POSIX file whose name really does
    contain a backslash still resolves correctly; the separator-agnostic
    reading is only consulted when that finds nothing.
    """
    candidates = [os.path.basename(stored_path)]
    agnostic = stored_path.replace('\\', '/').rsplit('/', 1)[-1]
    if agnostic and agnostic not in candidates:
        candidates.append(agnostic)
    return [c for c in candidates if c]


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

    A **remote URL is returned unchanged**. There is nothing on this filesystem
    to locate, and the fallbacks below would mangle it into a bogus local path,
    fail, and report the dependency as missing — a permanent miss for every
    statement that reads object storage. Handling it here rather than at each
    call site is deliberate: there are nine of them across restore, freshness,
    re-execution planning and virtual lineage, and a rule that nine callers must
    remember is a rule that will be forgotten. This makes them all correct by
    construction.
    """
    if is_remote_url(stored_path):
        return stored_path
    if os.path.exists(stored_path):
        return stored_path

    # Fallback 1: basename in CWD.
    #
    # ``os.path.basename`` only understands the HOST separator, so on POSIX a
    # path stored on Windows ("C:\\proj\\data.csv") has no recognisable
    # basename at all and this fallback silently resolves nothing — the exact
    # cross-platform case this function exists to survive. Fallback 2 below
    # already normalises separators; this one has to as well.
    #
    # Try the host interpretation first: on POSIX a filename may legitimately
    # contain a backslash, and that reading must keep winning.
    for basename in _basename_candidates(stored_path):
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


# Windows denies a replace whose destination is open; POSIX never does.
# Escalating 5/10/20/40/80/160ms -- ~315ms of total patience. Measured
# holders released on the first 10ms retry, so this is mostly headroom for a
# scanner that grabbed the file a moment longer.
REPLACE_RETRY_DELAYS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)


def replace_with_retry(tmp_path: str, path: str,
                       delays: tuple[float, ...] = REPLACE_RETRY_DELAYS) -> None:
    """``os.replace``, but tolerant of a destination that is briefly locked.

    The replace is atomic on both platforms, but on Windows it is not always
    *permitted*: if any handle currently has the destination open, the call
    fails with ``ERROR_ACCESS_DENIED`` rather than waiting. POSIX just swaps
    the directory entry and lets the reader finish on the old inode.

    Shared rather than duplicated because two call sites need it and they sit
    on opposite sides of the backend/notebook boundary -- the file backend
    (where it was first measured: a WinError 5 on effectively every Windows CI
    job and 10 of 12 consecutive local runs of one test, each silently
    discarding a cache entry) and ``notebook.loop_split.LoopSplitStore``,
    which had the same tmp-then-replace shape and swallowed the failure at
    debug level.

    A persistent denial (a read-only file, a genuinely stuck handle) still
    raises once the budget is spent: this waits out contention, it does not
    paper over a real permission problem.
    """
    for delay in delays:
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(tmp_path, path)      # out of patience; let it raise


@functools.lru_cache(maxsize=8)
def _module_stem(path: str) -> str:
    """The module name a file would have if it were imported."""
    return os.path.splitext(os.path.basename(path))[0]


def resolve_main_module(func: Any) -> str:
    """What to call ``__main__`` when qualifying *func* for a cache key.

    A function defined in the script you ran belongs to module ``__main__``,
    so ``python model.py`` keyed it as ``__main__.work`` while ``import model``
    keyed the same function, same source, same arguments as ``model.work``.
    Two entries, one computation -- and the common shape is exactly that:
    develop a script behind an ``if __name__ == "__main__"`` block, run it
    while testing, then import it from a driver and recompute everything.

    Resolving through ``__file__`` to the name the module would have on import
    makes those two agree, and it strictly REDUCES collisions on the other
    axis: today every script alike is ``__main__``, so two unrelated scripts
    with a same-named function meet; afterwards only two scripts with the same
    FILENAME do. (They still separate on source, helpers and read globals --
    the module name is a coarse guard on top of the state hash, not the thing
    doing the work.)

    Read from the FUNCTION's own globals, not ``sys.modules['__main__']``.
    Those are the same file for an ordinary ``python model.py``, and they are
    not under ``runpy`` or ``exec``, where the entry point is one file and the
    module claiming ``__main__`` is another -- taking the entry point there
    names the function after a file it was not defined in.

    Shared by ``Cash._get_func_key`` and the purity analyzer's
    ``_qualname_of``. Both feed the same cache key from different directions,
    and normalising only one of them leaves the state hash disagreeing between
    a direct run and an import while the function name agrees -- which is
    exactly the half-fixed state this function exists to prevent.

    Returns ``__main__`` unchanged when there is no ``__file__``: a REPL,
    ``python -c``, a frozen app, and a Jupyter kernel, where ``__main__`` is
    the user namespace rather than a file and there is no import to agree with.
    """
    path = (getattr(func, '__globals__', None) or {}).get('__file__')
    if not isinstance(path, str) or not path:
        return '__main__'
    return _module_stem(path) or '__main__'
