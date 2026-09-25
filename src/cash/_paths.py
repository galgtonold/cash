"""Path helpers: portable spelling, relocation, atomic replace, script names.

`normalize_path` and `is_remote_url` decide how a path is recorded in a key;
`resolve_file_dep_path` finds a recorded file after the project moved;
`replace_with_retry` is ``os.replace`` that waits out a briefly locked
destination on Windows; `resolve_main_module` names a script's module the way
an import would.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

__all__ = [
    "MAIN_MODULE_NAMES",
    "is_remote_url",
    "normalize_path",
    "replace_with_retry",
    "resolve_file_dep_path",
    "resolve_main_module",
]

# ``file://`` is excluded: it names a local path that can genuinely be stat'ed.
# A scheme of two characters or more: a single letter is a Windows drive, and
# ``f"{ROOT}/x.csv"`` with ``ROOT = "D:/"`` spells ``D://x.csv``, which Windows
# opens as a local file. Read as a URL with scheme ``d``, the read needed
# fsspec to be tracked at all, and raised without it.
_URL_SCHEME_RE = re.compile(r"^(?!file://)[a-zA-Z][a-zA-Z0-9+.\-]+://")


def is_remote_url(path: str) -> bool:
    """Whether *path* is a remote URL rather than a local filesystem path.

    A single definition, because the answer decides behaviour in a dozen
    places: whether the file tracker records a read on its remote channel,
    whether a dependency entry is validated by stat or by the store's
    validator, and whether a stored key can be path-resolved at all.
    """
    return bool(_URL_SCHEME_RE.match(path))


#: Windows' verbatim (``\\\\?\\``) and device (``\\\\.\\``) prefixes, in either
#: spelling of the separator.
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "//?/", "//./")


def normalize_path(path: str) -> str:
    """Return *path* with all OS-native separators replaced by forward slashes.

    Used to produce portable, platform-independent path strings for cache keys
    and dependency tracking.  On POSIX systems this is a no-op; on Windows it
    converts backslashes to forward slashes -- except under a verbatim or
    device prefix (``\\\\?\\C:\\...``), which is kept in backslashes.

    Examples::

        normalize_path("C:\\\\Users\\\\foo\\\\bar.csv")  # → "C:/Users/foo/bar.csv"
        normalize_path("/home/foo/bar.csv")              # → "/home/foo/bar.csv"
    """
    return _normalize_for(path, os.path.sep)


def _normalize_for(path: str, sep: str) -> str:
    """`normalize_path` for a platform whose separator is *sep*.

    Windows reads ``\\\\?\\`` only with backslashes: that prefix is how a path
    longer than 260 characters is opened at all, and it passes the rest to the
    file system as written. Rewritten as ``//?/C:/...``, the 260-character
    limit applied again, the recorded dependency could not be stat'ed, and
    the entry was stored without it -- an edit was then served stale.
    """
    if sep == "/":
        return path
    if path.startswith(_DEVICE_PREFIXES):
        return path.replace("/", sep)
    return path.replace(sep, "/")


def _basename_candidates(stored_path: str) -> list[str]:
    """Basenames to try for *stored_path*, most-trusted first.

    The host reading comes first so a POSIX file whose name really does
    contain a backslash still resolves correctly; the separator-agnostic
    reading is only consulted when that finds nothing.
    """
    candidates = [os.path.basename(stored_path)]
    agnostic = stored_path.replace("\\", "/").rsplit("/", 1)[-1]
    if agnostic and agnostic not in candidates:
        candidates.append(agnostic)
    return [c for c in candidates if c]


def resolve_file_dep_path(stored_path: str) -> str | None:
    """Find a recorded file dependency, also after the project moved.

    Tries the stored path, then its basename in the current directory, then
    ever longer suffixes of it (``examples/data.csv``) under the current
    directory. Returns the path found, or ``None``.

    A remote URL is returned unchanged: there is nothing local to find, and
    the fallbacks would turn it into a local path that does not exist, which
    reads as a missing dependency. Every caller relies on this.
    """
    if is_remote_url(stored_path):
        return stored_path
    if os.path.exists(stored_path):
        return stored_path

    # The basename in the current directory, read with either separator: a
    # path stored on Windows has no basename to `os.path.basename` on POSIX.
    for basename in _basename_candidates(stored_path):
        cwd_candidate = os.path.join(os.getcwd(), basename)
        if os.path.exists(cwd_candidate):
            return normalize_path(os.path.realpath(cwd_candidate))

    # Fallback 2: try progressively longer path suffixes relative to CWD.
    # E.g. stored = "C:/old/root/project/examples/data.csv"
    #   → try "examples/data.csv" relative to CWD
    parts = stored_path.replace("\\", "/").split("/")
    # Start from the second-to-last component (parent dir + filename)
    for i in range(max(len(parts) - 2, 1), 0, -1):
        suffix = "/".join(parts[i:])
        candidate = os.path.join(os.getcwd(), suffix)
        if os.path.exists(candidate):
            return normalize_path(os.path.realpath(candidate))

    return None


# Windows denies a replace whose destination is open; POSIX never does.
# Escalating waits, about 0.3 s in all.
REPLACE_RETRY_DELAYS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)


def replace_with_retry(tmp_path: str, path: str, delays: tuple[float, ...] = REPLACE_RETRY_DELAYS) -> None:
    """``os.replace`` that waits out a destination another process briefly holds.

    On Windows a replace fails with ``PermissionError`` while any handle has
    the destination open (a reader, a virus scanner). This retries with
    *delays*, then makes a last attempt that raises: a destination that stays
    locked is a real permission problem.
    """
    for delay in delays:
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(tmp_path, path)  # out of patience; let it raise


#: The names a script's own module goes by. ``__mp_main__`` is the script
#: re-imported in a multiprocessing child under the spawn start method (the
#: default on Windows and macOS), which must key its functions like the parent.
MAIN_MODULE_NAMES = frozenset({"__main__", "__mp_main__"})


def resolve_main_module(func: Any) -> str:
    """What to call ``__main__`` when qualifying *func* for a cache key.

    A function in the script you run belongs to ``__main__``, while the same
    function reached by ``import model`` belongs to ``model``. Naming the
    script's module as an import would name it makes both runs share entries.
    Read from the function's own globals, not ``sys.modules["__main__"]``:
    under ``runpy`` or ``exec`` those are different modules.

    ``Cash.get_func_key`` and the purity analyzer both use this, so the
    function name and the state hash agree between a direct run and an
    import. Returns ``__main__`` when there is no ``__file__`` (a REPL,
    ``python -c``, a notebook kernel).
    """
    g = getattr(func, "__globals__", None) or {}
    # `python -m pkg.mod` runs as `__main__`; its spec carries the dotted name
    # the import uses.
    spec_name = getattr(g.get("__spec__"), "name", None)
    if isinstance(spec_name, str) and spec_name and spec_name not in MAIN_MODULE_NAMES:
        return spec_name
    path = g.get("__file__")
    if not isinstance(path, str) or not path:
        return "__main__"
    return os.path.splitext(os.path.basename(path))[0] or "__main__"
