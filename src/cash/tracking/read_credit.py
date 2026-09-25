"""Crediting a read to the code that made it, for memos that outlive a call.

A memo (``functools.lru_cache``, a module dict) hands a later call what an
earlier read produced, and the later call reads nothing. So each read is also
credited to the user code on the stack, and `credited_reads` tells a cached
call which files the helpers it reached have read, and which version of each.
Reads made outside every cached call count too (`note_untracked_read`).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

from cash._memo import CODE_OBJECTS, READ_PATHS, SOURCE_FILES, LruMemo
from cash._paths import is_remote_url, normalize_path
from cash.install_paths import is_user_path
from cash.tracking.read_classification import is_cash_internal, is_pseudo_fs, regular_file_stat

if TYPE_CHECKING:
    from cash.tracking.file_tracker import FileAccessTracker

__all__ = ["credit_read_to_stack", "credited_reads", "note_untracked_read"]

logger = logging.getLogger(__name__)


#: ``code -> {file: its stat when last read}`` for files read while a frame of
#: that code was on the stack, process-wide. A memo (``functools.lru_cache``, a
#: module dict) hands a later call the product of an earlier read, and the
#: later call reads nothing -- so its entry recorded no file and kept serving
#: after the file changed. What a helper read once is what
#: `credited_reads` answers when a call reaches it again, and the stat says
#: WHICH version it read: a memo filled before the file changed hands back the
#: old version's data. Reads outside any cached call count too
#: (`note_untracked_read`) -- `main()` logging its settings through the memo
#: before the first cached call is the ordinary way to fill one. A code past
#: `_READS_PER_CODE_MAX` files is marked ``None``: it reads per argument, and
#: every file it ever read is no one call's dependency.
_READS_PER_CODE_MAX = 16
_reads_by_code: LruMemo[Any, dict[str, Any] | None] = LruMemo(CODE_OBJECTS)
_CASH_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _record_read(code: Any, abs_path: str, stat: Any) -> None:
    """Remember that *code* read *abs_path*, as it was (*stat*)."""
    memo = _reads_by_code
    reads = memo.get(code, ())
    if reads is None:
        return
    if code not in memo:
        reads = {}
        memo[code] = reads
    if abs_path not in reads and len(reads) >= _READS_PER_CODE_MAX:
        memo[code] = None
    else:
        reads[abs_path] = stat  # the LATEST read: a memo refilled is current again


#: The decorator's code: `Cash` in core.py and the call steps in decorator/.
_DECORATOR_FILES = (
    os.path.normcase(os.path.join(_CASH_PACKAGE_DIR, "core.py")),
    os.path.normcase(os.path.join(_CASH_PACKAGE_DIR, "decorator")) + os.sep,
)


def _is_cash_wrapper(filename: str) -> bool:
    norm = os.path.normcase(filename)
    return norm == _DECORATOR_FILES[0] or norm.startswith(_DECORATOR_FILES[1])


def credit_read_to_stack(abs_path: str, tracker: "FileAccessTracker") -> None:
    """Credit a read to the user code on the stack, up to the cached call."""
    try:
        frame = sys._getframe(1)
    except ValueError:
        return
    stat = tracker.read_stats.get(abs_path)
    depth = 0
    while frame is not None and depth < 64:
        code = frame.f_code
        kind = frame_kind(code.co_filename)
        if kind == "wrapper":
            break  # the cached call's own wrapper: the walk ends
        if kind == "user":
            tracker.note_reading_code(code)
            _record_read(code, abs_path, stat)
        frame, depth = frame.f_back, depth + 1


#: code filename -> ``wrapper``/``cash``/``user``/``other``; `frame_kind`.
_frame_kinds: LruMemo[str, str] = LruMemo(SOURCE_FILES)


def frame_kind(filename: str) -> str:
    """``wrapper`` (the cached call's own), ``cash``, ``user`` or ``other``,
    remembered per filename: every read walks the stack."""
    kind = _frame_kinds.get(filename)
    if kind is None:
        kind = (
            "wrapper"
            if _is_cash_wrapper(filename)
            else "cash"
            if filename and os.path.normcase(filename).startswith(_CASH_PACKAGE_DIR)
            else "user"
            if is_user_path(filename)
            else "other"
        )
        _frame_kinds[filename] = kind
    return kind


#: absolute path -> resolved path, for reads outside a tracker; `note_untracked_read`.
_untracked_realpaths: LruMemo[str, str] = LruMemo(READ_PATHS)
#: resolved path -> (monotonic time, stat); `note_untracked_read`.
_untracked_stats: LruMemo[str, tuple[float, Any]] = LruMemo(READ_PATHS)


def note_untracked_read(path: Any, frame: Any) -> None:
    """A read made outside every cached call, credited to the user code on the stack.

    *frame* is the frame that asked for the read. Only a read that user code
    started: a read by cash itself or by a library with no user frame above it
    (an import, a font cache) is nobody's input. Never raises -- it runs inside
    every ``open`` in the process.
    """
    try:
        codes = []
        depth = 0
        while frame is not None and depth < 64:
            kind = frame_kind(frame.f_code.co_filename)
            if kind == "wrapper":
                break
            if kind == "cash" and not codes:
                return  # cash reading its own files
            if kind == "user":
                codes.append(frame.f_code)
            frame, depth = frame.f_back, depth + 1
        if not codes:
            return
        raw = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
        if not isinstance(raw, str) or is_pseudo_fs(raw) or is_remote_url(raw):
            return
        # `realpath` is 60us on Windows, most of what this costs; resolved once
        # per absolute path (so a chdir still resolves anew).
        absolute = os.path.abspath(raw)
        abs_path = _untracked_realpaths.get(absolute)
        if abs_path is None:
            abs_path = normalize_path(os.path.realpath(absolute))
            _untracked_realpaths[absolute] = abs_path
        if is_pseudo_fs(abs_path) or is_cash_internal(abs_path):
            return
        # A stat is 15us, and a loop re-reading one file pays it every time.
        # Reusing one taken in the last second can only be too OLD, and an old
        # stat that differs from the file makes a store refused, never a stale
        # answer served (`FileDeps.credit_remembered_reads`).
        now = time.monotonic()
        seen = _untracked_stats.get(abs_path)
        if seen is not None and now - seen[0] < 1.0:
            stat = seen[1]
        else:
            stat = regular_file_stat(abs_path)
            _untracked_stats[abs_path] = (now, stat)
        if stat is None:
            return
        for code in codes:
            _record_read(code, abs_path, stat)
    except Exception:  # attribution is an aid, never a failure
        logger.debug("[TRACKER] could not note an untracked read of %r", path, exc_info=True)


def credited_reads(code: Any) -> dict[str, Any] | None:
    """``{file: stat when read}`` for files read while *code* was on the stack;
    None when it reads per argument."""
    reads = _reads_by_code.get(code, ())
    return None if reads is None else dict(reads)
