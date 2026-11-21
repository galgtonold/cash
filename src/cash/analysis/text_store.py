"""Answers worked out from a source text alone, kept on disk, so a fresh
process does not work them out again for every helper its cached functions
reach.

A ``@cash.cache`` function's first call in a new process, its purity report
read back from `report_store`, still parsed and walked the source of every
helper: to digest it as a key (`source_norm.source_identity_digest`), for
the globals its body may change (`GlobalReads.read_global_data_names`), for
the imports in its body (`GlobalReads.local_binding_plan`) and for the names
its annotations read (`hit_memo`). Each is a function of the text and of a
few small inputs its caller names (*extra*), so each is stored under a
digest of all of them, and an answer is used only when that digest is the
same:

* the text is digested in full, on every lookup, so an edit, a revert, a
  moved or copied function compare by what is in them;
* *extra* spells every other input the answer reads (the package a relative
  import resolves in, the names folded, a code object's free names);
* the Python that parses the text and cash's own code are the store's
  folder (`scan_store.store_dir`).

Nothing that depends on what a name is bound to, a file other than the one
the text is, or an installed library is stored here: a caller whose answer
reads any of that refuses to store it (its *encode* raises ValueError).

One file per kind, read once per process, written at exit and in batches,
merged with what another process wrote meanwhile; the answers this process
used are kept first when a file is full. A file that cannot be read, or an
answer that does not decode, is a miss. ``CASH_ANALYSIS_STORE=0`` turns it
off with the other stores; ``CASH_ANALYSIS_MEMO_VERIFY=1`` works out every
stored answer used again and raises `StoredTextMismatch` when it differs.
"""

from __future__ import annotations

import atexit
import hashlib
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from . import scan_store

__all__ = ["STATS", "StoredTextMismatch", "flush", "held", "kept", "status"]

T = TypeVar("T")

#: Layout of a stored file; a change makes every older one a miss.
_FORMAT = 1
_KIND = "text-answers"
#: Answers kept per kind.
_MAX_ANSWERS = 32768
#: Answers worked out before they are written (the rest at exit).
_FLUSH_EVERY = 4096

STATS = {"loaded": 0, "computed": 0, "stored": 0, "refused": 0, "verified": 0}

_LOCK = threading.RLock()
#: kind -> {digest: answer}: what this process read or worked out.
_SLOTS: dict[str, dict[str, Any]] = {}
#: kind -> digests worked out here and not written yet.
_PENDING: dict[str, set[str]] = {}
#: kind -> digests used in this process (kept last, so first, when full).
_USED: dict[str, set[str]] = {}
#: kind -> the folder its file was read from: what this process works out
#: goes back there, where the next one reads it (`flush`).
_DIRS: dict[str, Any] = {}


class StoredTextMismatch(AssertionError):
    """``CASH_ANALYSIS_MEMO_VERIFY=1``: a stored answer is not what working
    it out again from the text gives."""


def status() -> dict:
    return dict(STATS)


def _file(directory: Any, kind: str) -> Any:
    slot = hashlib.blake2b(f"{_KIND}\0{kind}".encode(), digest_size=12).hexdigest()
    return directory / f"{slot}.json"


def _read_file(directory: Any, kind: str) -> dict[str, Any]:
    data = scan_store.read_json(_file(directory, kind))
    if not isinstance(data, dict) or data.get("kind") != _KIND or data.get("format") != _FORMAT:
        return {}
    if data.get("of") != kind:
        return {}
    answers = data.get("answers")
    return dict(answers) if isinstance(answers, dict) else {}


def _slot(kind: str) -> dict[str, Any] | None:
    slot = _SLOTS.get(kind)
    if slot is None:
        directory = scan_store.store_dir()
        if directory is None:
            return None
        slot = _SLOTS[kind] = _read_file(directory, kind)
        _DIRS[kind] = directory
    return slot


def held(kind: str) -> bool:
    """Has this process read *kind*'s file (or found there is none to read)?
    A caller whose answer is cheaper to work out than that file is to read
    works small ones out until it is."""
    return kind in _SLOTS


def _digest(kind: str, text: str, extra: str) -> str:
    h = hashlib.blake2b(digest_size=20)
    h.update(f"{kind}\0{len(extra)}\0{extra}\0".encode("utf-8", "surrogatepass"))
    h.update(text.encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def kept(
    kind: str,
    text: str,
    extra: str,
    compute: Callable[[], T],
    encode: Callable[[T], Any],
    decode: Callable[[Any], T],
) -> T:
    """``compute()``, from the store when an earlier process stored it for
    this *text* and *extra* under this Python and this cash. *compute* must
    read nothing but *text* and what *extra* spells. *encode* turns its
    answer to JSON, or raises ValueError when it must not be stored;
    *decode* raises ValueError (or a lookup or type error) on a stored
    answer that does not fit: it is worked out again."""
    if not scan_store.enabled():
        return compute()
    try:
        with _LOCK:
            slot = _slot(kind)
    except OSError:
        slot = None
    if slot is None:
        return compute()
    digest = _digest(kind, text, extra)
    if digest in slot:
        try:
            answer = decode(slot[digest])
        except (ValueError, TypeError, LookupError, AttributeError):
            pass
        else:
            STATS["loaded"] += 1
            with _LOCK:
                _USED.setdefault(kind, set()).add(digest)
            from . import tree_memo

            if tree_memo.VERIFY:
                STATS["verified"] += 1
                if compute() != answer:
                    raise StoredTextMismatch(f"stored {kind} answer differs from working it out again")
            return answer
    answer = compute()
    STATS["computed"] += 1
    try:
        encoded = encode(answer)
    except (ValueError, TypeError):
        STATS["refused"] += 1
        return answer
    with _LOCK:
        slot[digest] = encoded
        _PENDING.setdefault(kind, set()).add(digest)
        _USED.setdefault(kind, set()).add(digest)
        due = sum(len(p) for p in _PENDING.values()) >= _FLUSH_EVERY
    if due:
        flush()
    return answer


def flush(*, at_exit: bool = False) -> None:
    """Write what this process worked out since the last flush, merged into
    what is on disk now. Best effort.

    Each kind goes back to the folder its file was read from (`_DIRS`): a
    kind first asked for in a call of a ``Cash(cache_dir=...)`` reads that
    instance's folder, and at exit, with no call running, the folder was
    the default instance's when one existed; the next process read the
    user's folder again, found nothing, and worked out the same answers
    (about 580 of them for an lmfit fit's first call)."""
    with _LOCK:
        if not _PENDING:
            return
        pending = dict(_PENDING)
        _PENDING.clear()
        try:
            default = scan_store.exit_store_dir() if at_exit else scan_store.store_dir()
        except Exception:  # noqa: BLE001 - at exit: a store that cannot be found keeps nothing
            default = None
        for kind, digests in pending.items():
            directory = _DIRS.get(kind)
            if directory is None or not scan_store.still_there(directory):
                directory = default
            if directory is None:
                continue
            mine = _SLOTS.get(kind, {})
            merged = _read_file(directory, kind)
            merged.update((d, mine[d]) for d in digests if d in mine)
            used = _USED.get(kind, set())
            ordered = {d: a for d, a in merged.items() if d not in used}
            ordered.update((d, a) for d, a in merged.items() if d in used)
            answers = dict(list(ordered.items())[-_MAX_ANSWERS:])
            data = {"kind": _KIND, "format": _FORMAT, "of": kind, "answers": answers}
            if scan_store.write_json(directory, _file(directory, kind), data):
                STATS["stored"] += 1


def forget() -> None:
    """Drop what this process read or worked out (tests that edit the store)."""
    with _LOCK:
        _SLOTS.clear()
        _PENDING.clear()
        _USED.clear()
        _DIRS.clear()


atexit.register(flush, at_exit=True)
