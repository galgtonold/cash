"""The identity of code: a function's own source, the classes and
callables it reaches, and which code counts as the user's."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning

logger = logging.getLogger(__name__)


# id(code object) -> (the object itself, its source digest). Keyed by IDENTITY,
# and the object is retained so the id cannot be recycled under us -- the same
# guard `function_tracker._source_cache` uses.
#
# NOT keyed on the code object directly, which was the first attempt: CodeType
# implements __eq__/__hash__ BY VALUE, and co_filename is not part of that
# equality, so two helpers with the same body in different modules share one
# dict slot. That made `test_real_helper_change_still_recomputes` fail
# reproducibly under xdist while passing alone -- a stale digest served across
# tests through a module-level memo.
#
# A redefinition (reloaded module, re-run cell) compiles a NEW code object, so
# identity keying still cannot serve a digest for code that is no longer
# running. Editing a .py file WITHOUT reloading leaves the old code object
# live, and the old digest is then the correct answer.
#
# Load-bearing, not a micro-optimisation. `_hash_callable_source` is the live
# per-call identity of every transitive helper, and it calls
# `inspect.getsource`, which re-reads and RE-TOKENISES the source block on
# every call. Measured on a 2-helper function: 8700 tokenizer calls per 300
# cache hits, and 37ms of a 65ms key computation.
#
# Module-level rather than per-instance: the digest depends only on the code
# object, so two Cash instances cannot legitimately disagree about it.
SOURCE_HASH_MEMO: dict = {}
SOURCE_HASH_MEMO_MAX = 4096
#: ``id(code) -> (code, path, size, mtime_ns, text digest)``: the stat of the
#: file whose text a function's key was read from, taken just before reading
#: it, and that text's digest. The store compares both with the file now
#: (`Cash._code_moved_since_keyed`).
CODE_KEYED_STATS: dict[int, tuple[Any, str, int, int, str]] = {}


#: Source files already reported as edited-since-load, one notice per file.
_SOURCE_CHANGED_WARNED: set[str] = set()


def stat_code_file(fn: Any) -> tuple[Any, str, int, int] | None:
    """``(code, path, size, mtime_ns)`` for *fn*'s source file, or None."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if not path or path.startswith("<"):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (code, path, st.st_size, st.st_mtime_ns)


def warn_source_changed_since_load(fn: Callable) -> None:
    """Say, once per file, that a helper is keyed by its loaded code."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if path in _SOURCE_CHANGED_WARNED:
        return
    _SOURCE_CHANGED_WARNED.add(path)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "a function")
    try:
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "KEY-SOURCE-CHANGED",
            f"{path} was edited after this process loaded it, so the code running "
            f"{name}() is the old version while the file holds a new one. cash "
            f"keys it by the code actually running, so results stay correct for "
            f"this process -- but they are not the new code's results, and they "
            f"will not be reused once the process restarts.",
            "restart the process to run the new code. If a deploy puts new files "
            "on disk before the restart, this is the window it opens.",
        )
    except Exception:  # noqa: BLE001 - a notice must never break a call
        logger.debug("Could not emit the source-changed notice", exc_info=True)


#: Pydantic v2 compiles these onto every model class. They are derived from the
#: field declarations and their digest differs in every process, so folding them
#: made a pydantic spec un-cacheable across runs. `Cash._pydantic_field_parts`
#: folds the declarations they were standing in for.
PYDANTIC_COMPILED = frozenset(
    {
        "__pydantic_core_schema__",
        "__pydantic_serializer__",
        "__pydantic_validator__",
    }
)
