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

from cash._memo import READ_PATHS, SOURCE_FILES, LruMemo
from cash._paths import is_remote_url, normalize_path
from cash.install_paths import is_user_path
from cash.tracking.read_classification import is_cash_internal, is_pseudo_fs, regular_file_stat

from .. import _weak

if TYPE_CHECKING:
    from cash.tracking.file_tracker import FileAccessTracker

__all__ = [
    "IDLE_READS_PER_CODE",
    "credit_read_to_stack",
    "credit_read_again",
    "credited_reads",
    "note_untracked_read",
    "reads_not_kept",
]

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
#: before the first cached call is the ordinary way to fill one. Every file a
#: code read is kept, however many: a loader memoised over twenty files hands
#: its consumer all twenty. A memo keyed by the path a call was given adds only
#: that path (`FileDeps.credit_remembered_reads`). Held weakly by the code,
#: not in a bounded memo: evicting a loader's record while its memo still
#: serves would leave the next consumer with no dependency at all, and the
#: record goes when nothing can run the code any more.
class _ReadsByCode(_weak.WeakKeyDictionary):
    """A `weakref.WeakKeyDictionary` that counts the entries it lost
    (``gone``): until one goes, what a lookup found is still the entry."""

    def __init__(self) -> None:
        super().__init__()
        self.gone = 0
        drop = self._remove

        def remove(k: Any, drop: Any = drop, selfref: Any = _weak.ref(self)) -> None:
            self = selfref()
            if self is not None:
                self.gone += 1
            drop(k)

        self._remove = remove


_reads_by_code: _ReadsByCode = _ReadsByCode()

#: ``id(code) -> (_reads_by_code.gone, ref to code, its reads)``: the entry a
#: lookup found, while no entry has gone since. A lookup hashes the code and
#: compares it with the key it finds, constants and all (~4 us for a cell's
#: code), and a loop over a folder's files credits each read to the same
#: few codes.
_found_reads: dict[int, tuple[int, Any, dict[str, Any]]] = {}
_CASH_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _record_read(code: Any, abs_path: str, stat: Any) -> None:
    """Remember that *code* read *abs_path*, as it was (*stat*)."""
    if _unresolved_by_code:
        _resolve_untracked(code)  # older than this read: in first
    memo = _reads_by_code
    found = _found_reads.get(id(code))
    if found is not None and found[0] == memo.gone and found[1]() is code:
        reads = found[2]
    else:
        reads = memo.get(code)
        if reads is None:
            reads = {}
            memo[code] = reads
        if len(_found_reads) >= _FOUND_READS_MAX:
            _found_reads.clear()
        _found_reads[id(code)] = (memo.gone, _weak.ref(code), reads)
    reads[abs_path] = stat  # the LATEST read: a memo refilled is current again


#: Codes `_found_reads` keeps at most.
_FOUND_READS_MAX = 256


#: The decorator's code: `Cash` in core.py and the call steps in decorator/.
_DECORATOR_FILES = (
    os.path.normcase(os.path.join(_CASH_PACKAGE_DIR, "core.py")),
    os.path.normcase(os.path.join(_CASH_PACKAGE_DIR, "decorator")) + os.sep,
)


def _is_cash_wrapper(filename: str) -> bool:
    norm = os.path.normcase(filename)
    return norm == _DECORATOR_FILES[0] or norm.startswith(_DECORATOR_FILES[1])


def credit_read_to_stack(abs_path: str, tracker: "FileAccessTracker") -> tuple[Any, ...]:
    """Credit a read to the user code on the stack, up to the cached call;
    the codes credited."""
    try:
        frame = sys._getframe(1)
    except ValueError:
        return ()
    stat = tracker.read_stats.get(abs_path)
    anchor = tracker.credit_anchor
    below = None  # the kind of the frame last walked
    credited: list[Any] = []
    while frame is not None:
        if anchor is not None and frame is anchor[0]:
            # Walked from here before: a frame's callers never change while
            # it runs, so the user code above it is what the last walk found.
            for code in anchor[1]:
                tracker.note_reading_code(code)
                _record_read(code, abs_path, stat)
            return (*credited, *anchor[1])
        code = frame.f_code
        kind = frame_kind(code.co_filename)
        if kind == "wrapper":
            break  # the cached call's own wrapper: the walk ends
        if kind == "cash" and below is not None and below != "cash":
            # cash's own frame above the user's or a library's (a notebook
            # statement's runner): every read the code below it makes walks
            # the same frames above it, dozens in a notebook.
            above = _user_codes_above(frame)
            tracker.credit_anchor = (frame, above)
            for code in above:
                tracker.note_reading_code(code)
                _record_read(code, abs_path, stat)
            return (*credited, *above)
        if kind == "user":
            tracker.note_reading_code(code)
            _record_read(code, abs_path, stat)
            credited.append(code)
        below = kind
        frame = frame.f_back
    return tuple(credited)


def credit_read_again(abs_path: str, tracker: "FileAccessTracker", codes: tuple[Any, ...]) -> None:
    """Credit a read to *codes*, the ones `credit_read_to_stack` credited the
    last read from the same frames (`FileAccessTracker` checks that)."""
    stat = tracker.read_stats.get(abs_path)
    for code in codes:
        _record_read(code, abs_path, stat)


def _user_codes_above(frame: Any) -> tuple[Any, ...]:
    """The user code at *frame* and above it, up to the cached call's wrapper."""
    codes = []
    while frame is not None:
        kind = frame_kind(frame.f_code.co_filename)
        if kind == "wrapper":
            break
        if kind == "user":
            codes.append(frame.f_code)
        frame = frame.f_back
    return tuple(codes)


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
#: absolute path -> (monotonic time, stat); `note_untracked_read`.
_untracked_stats: LruMemo[str, tuple[float, Any]] = LruMemo(READ_PATHS)

#: How many files read outside every cached call are remembered for one
#: code. A memoised loader reads a handful; a program that walks a folder
#: of images with cash merely imported reads without end, and each read is
#: on the stack of the same few functions. A code that reaches this stops
#: being recorded and is marked (`reads_not_kept`): what it read after that
#: is unknown, so a cached call that later reaches it without reading
#: through it is not stored, with a warning, where a silent cap would let
#: it depend on nothing. Once every function on a read's stack is so
#: marked, the read costs a walk of the stack and nothing else.
IDLE_READS_PER_CODE = READ_PATHS
#: The same, over all codes: past it the code remembering the most gives up
#: its older half, and is marked too.
IDLE_READS_TOTAL = 4 * READ_PATHS


class _IdleReads(dict):
    """``{absolute path: stat when read}`` for one code's reads outside every
    cached call, the latest last, not resolved yet.

    Kept by the code's ``id`` with a weak reference to it, not in a
    dictionary keyed by the code: looking a code up hashes its constants and
    names, ~1.5 us for each function on the stack of every ``open`` in a
    process that imported cash."""

    __slots__ = ("code", "lost", "moved")

    def __init__(self, code: Any) -> None:
        super().__init__()
        key = id(code)
        twins = _idle_twins.setdefault(hash(code), [])  # hashed once per code, not per read
        twins.append(key)

        def gone(ref: Any) -> None:
            found = _unresolved_by_code.get(key)
            if found is not None and found.code is ref:
                _idle_total[0] -= len(found)
                del _unresolved_by_code[key]
            try:
                twins.remove(key)
            except ValueError:
                pass
            if not twins and _idle_twins.get(hash_of_code) is twins:
                del _idle_twins[hash_of_code]

        hash_of_code = hash(code)
        self.code = _weak.ref(code, gone)
        #: Reads of this code went unremembered (`IDLE_READS_PER_CODE`).
        self.lost = False
        #: How many of its reads `_resolve_untracked` has moved on.
        self.moved = 0


#: ``id(code) -> its reads outside a cached call`` (`_IdleReads`). `realpath`
#: is ~60 us on Windows and was most of what every ``open`` in a process
#: that imported cash cost; a path is resolved once something asks what the
#: code read (`credited_reads`), or before the code's next read inside a
#: cached call is recorded, so the latest read of a file still wins.
_unresolved_by_code: dict[int, _IdleReads] = {}
#: How many paths `_unresolved_by_code` holds, over all codes.
_idle_total = [0]
#: ``hash(code) -> ids of the codes with a record``: two equal codes (one
#: source compiled twice) share what they read, as in `_reads_by_code`.
_idle_twins: dict[int, list[int]] = {}


def _idle_reads(code: Any) -> _IdleReads | None:
    """*code*'s own record in `_unresolved_by_code`, or None."""
    found = _unresolved_by_code.get(id(code))
    if found is not None and found.code() is code:
        return found
    return None


def _idle_records(code: Any) -> list[_IdleReads]:
    """The records of *code* and of every live code equal to it."""
    found = []
    for key in _idle_twins.get(hash(code), ()):
        record = _unresolved_by_code.get(key)
        twin = record.code() if record is not None else None
        if twin is not None and (twin is code or twin == code):
            found.append(record)
    return found


def _trim_idle() -> None:
    """Take the older half of the largest record away (`IDLE_READS_TOTAL`)."""
    largest = max(_unresolved_by_code.values(), key=len, default=None)
    if largest is None or not largest:
        return
    drop = (len(largest) + 1) // 2
    for absolute in list(largest)[:drop]:
        largest.pop(absolute, None)
    largest.lost = True
    _idle_total[0] -= drop


def _resolve(absolute: str) -> str:
    resolved = _untracked_realpaths.get(absolute)
    if resolved is None:
        resolved = normalize_path(os.path.realpath(absolute))
        _untracked_realpaths[absolute] = resolved
    return resolved


def _resolve_untracked(code: Any, pending: _IdleReads | None = None) -> None:
    """Move *code*'s unresolved reads into `_reads_by_code`, in the order
    they were made, counting the files that are new there
    (`_IdleReads.moved`): a code asked about between its reads would
    otherwise keep every file after all."""
    if pending is None:
        pending = _idle_reads(code)
    if not pending:
        return
    items = list(pending.items())
    _idle_total[0] -= len(items)
    pending.clear()
    memo = _reads_by_code
    reads = memo.get(code)
    if reads is None:
        reads = {}
        memo[code] = reads
    for absolute, stat in items:
        try:
            abs_path = _resolve(absolute)
        except (OSError, ValueError):
            continue
        if is_pseudo_fs(abs_path) or is_cash_internal(abs_path):
            continue
        if abs_path not in reads:
            pending.moved += 1
        reads[abs_path] = stat


def reads_not_kept(code: Any) -> bool:
    """Did *code* read more files outside every cached call than cash
    remembers for it? Then `credited_reads` is not all it read."""
    if not _unresolved_by_code:
        return False
    return any(record.lost for record in _idle_records(code))


def note_untracked_read(path: Any, frame: Any) -> None:
    """A read made outside every cached call, credited to the user code on the stack.

    *frame* is the frame that asked for the read. Only a read that user code
    started: a read by cash itself or by a library with no user frame above it
    (an import, a font cache) is nobody's input. Never raises -- it runs inside
    every ``open`` in the process.
    """
    try:
        codes = []
        unresolved = _unresolved_by_code
        open_records = False  # is any code on the stack still recorded?
        while frame is not None:
            code = frame.f_code
            kind = frame_kind(code.co_filename)
            if kind == "wrapper":
                break
            if kind == "cash" and not codes:
                return  # cash reading its own files
            if kind == "user":
                codes.append(code)
                if not open_records:
                    pending = unresolved.get(id(code))
                    open_records = pending is None or not pending.lost or pending.code() is not code
            frame = frame.f_back
        if not open_records:
            return  # nobody's, or every reader past what cash remembers
        raw = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
        if not isinstance(raw, str) or is_pseudo_fs(raw) or is_remote_url(raw):
            return
        # By absolute path (so a chdir still resolves anew), resolved later
        # (`_unresolved_by_code`).
        absolute = os.path.abspath(raw)
        # A stat is 15us, and a loop re-reading one file pays it every time.
        # Reusing one taken in the last second can only be too OLD, and an old
        # stat that differs from the file makes a store refused, never a stale
        # answer served (`FileDeps.credit_remembered_reads`).
        now = time.monotonic()
        seen = _untracked_stats.get(absolute)
        if seen is not None and now - seen[0] < 1.0:
            stat = seen[1]
        else:
            stat = regular_file_stat(absolute)
            _untracked_stats[absolute] = (now, stat)
        if stat is None:
            return
        total = _idle_total
        for code in codes:
            pending = unresolved.get(id(code))
            if pending is None or pending.code() is not code:
                if pending is not None:
                    total[0] -= len(pending)  # a code gone, its id used again
                pending = unresolved[id(code)] = _IdleReads(code)
            elif pending.lost:
                continue
            elif absolute in pending:
                del pending[absolute]  # the latest read last
                total[0] -= 1
            elif len(pending) + pending.moved >= IDLE_READS_PER_CODE:
                pending.lost = True
                continue
            pending[absolute] = stat
            total[0] += 1
        if total[0] > IDLE_READS_TOTAL:
            _trim_idle()
    except Exception:  # attribution is an aid, never a failure
        logger.debug("[TRACKER] could not note an untracked read of %r", path, exc_info=True)


def credited_reads(code: Any) -> dict[str, Any]:
    """``{file: stat when read}`` for files read while *code* was on the stack."""
    if _unresolved_by_code:
        for record in _idle_records(code):
            _resolve_untracked(code, record)
    return dict(_reads_by_code.get(code) or {})
