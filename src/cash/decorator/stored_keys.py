"""The record of which keys earlier runs stored, kept beside the cache so a
new process can say why its first call missed.

One small JSON file per function under ``<cache>/.keys/``. Changes are kept
in memory and written by a background writer, several stores to one file at a
time and each file at most once per `StoredKeyRecord.WRITE_INTERVAL`; results
kept in RAM only and warnings shown ride along with the next write, or with
`StoredKeyRecord.flush` at shutdown.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .._memo import RECORDS, LruMemo
from .._paths import replace_with_retry
from ..backends._writes import PendingWrites
from ..backends.cache_dir import recreate_cache_dir
from ..tracking.tracker_context import untracked
from .call_state import PROCESS_STARTED

logger = logging.getLogger(__name__)

__all__ = ["StoredKeyRecord"]

#: The kinds a record holds. ``keys``: ``{cache_key: [stored_at, ttl]}`` for
#: results that reached disk. ``ram_only``: ``{cache_key: [computed_at, why]}``
#: for results kept in RAM only. ``states``: ``{state: flat ledger}``.
#: ``warned``: ``{digest: shown_at}`` of warnings already shown.
_KINDS = ("keys", "ram_only", "states", "warned")

Doc = dict[str, dict[str, Any]]


def _empty() -> Doc:
    return {kind: {} for kind in _KINDS}


def _put_last(entries: dict, key: str, value: Any) -> None:
    entries.pop(key, None)
    entries[key] = value


def _trim(entries: dict, most: int) -> None:
    while len(entries) > most:
        entries.pop(next(iter(entries)))


@dataclass
class _Changes:
    """What this process learned about one record and has not written yet."""

    func: str
    keys: dict[str, list] = field(default_factory=dict)
    ram_only: dict[str, list] = field(default_factory=dict)
    #: state -> its flat ledger, or None to only mark it most recent.
    states: dict[str, dict | None] = field(default_factory=dict)
    warned: dict[str, float] = field(default_factory=dict)

    def copy(self) -> _Changes:
        return _Changes(self.func, dict(self.keys), dict(self.ram_only), dict(self.states), dict(self.warned))

    def apply(self, doc: Doc) -> None:
        for key, value in self.keys.items():
            doc["ram_only"].pop(key, None)  # it reached disk after all
            _put_last(doc["keys"], key, value)
        for key, value in self.ram_only.items():
            doc["keys"].pop(key, None)  # its disk copy is gone: this run recomputed it
            _put_last(doc["ram_only"], key, value)
        for state, flat in self.states.items():
            known = doc["states"].pop(state, None)
            if known is not None or flat:
                doc["states"][state] = known if known is not None else flat
        doc["warned"].update(self.warned)


class MissFacts(NamedTuple):
    """What a record says about one key that was looked up and not found:
    the answers `MissHistory.absent_entry_reason` asks of it."""

    #: ``[stored_at, ttl]`` when an earlier run stored this key, else None.
    stored: list | None
    #: ``[computed_at, why]`` when a run kept it in RAM only, else None.
    ram_only: list | None
    #: The newest key with the same dynamic and argument parts under another
    #: state (results kept in RAM only first, as the record is read newest
    #: first), or None.
    same_arguments: str | None
    #: The states of the entries an earlier process left.
    earlier_states: frozenset
    #: The newest of those entries, when the key's state is not among them.
    newest_earlier: str | None
    #: The newest stored key that is not this one, or None.
    last_other: str | None


def _key_parts(key: str) -> list[str]:
    return key.rsplit(":", 3)


def _earlier_time(value: Any) -> float | None:
    """When an entry an EARLIER process wrote was written, else None."""
    if value and isinstance(value[0], (int, float)) and value[0] < PROCESS_STARTED:
        return value[0]
    return None


def facts_from_doc(doc: Doc, cache_key: str) -> MissFacts:
    """`MissFacts` for *cache_key*, read from a whole record (`StoredKeyRecord.read`).

    The reference for `_View.facts`, which answers the same from indexes."""
    record, ram_only = doc["keys"], doc["ram_only"]
    new = _key_parts(cache_key)
    same = None
    earlier_states: frozenset = frozenset()
    newest = None
    if len(new) == 4:
        for key in reversed([*record, *ram_only]):
            old = _key_parts(key)
            if len(old) == 4 and old[2:] == new[2:] and old[1] != new[1]:
                same = key
                break
        earlier = {
            key: value
            for kind in ("keys", "ram_only")
            for key, value in doc[kind].items()
            if _earlier_time(value) is not None
        }
        earlier_states = frozenset(key.rsplit(":", 3)[1] for key in earlier if key.count(":") >= 3)
        if earlier_states and new[1] not in earlier_states:
            newest = max(earlier, key=lambda key: earlier[key][0])
    others = [key for key in record if key != cache_key]
    return MissFacts(
        record.get(cache_key), ram_only.get(cache_key), same, earlier_states, newest, others[-1] if others else None
    )


class _View:
    """One record as `StoredKeyRecord.read` would return it -- the file plus
    the changes not written yet -- kept up to date as keys are noted, with
    the indexes a miss reason asks. A miss of a cheap function read and
    merged the whole record, then walked every key in it, on every call."""

    def __init__(self, disk: Doc | None, overlays: list[_Changes]) -> None:
        #: The disk record this view was built on (`StoredKeyRecord._memo`), by identity.
        self.disk = disk
        self.keys: dict[str, Any] = {}
        self.ram_only: dict[str, Any] = {}
        #: kind -> (dynamic part, argument part) -> {key: state}, in record order.
        self._by_args: dict[str, dict[tuple[str, str], dict[str, str]]] = {"keys": {}, "ram_only": {}}
        #: key -> (written at, state or None) for the entries an earlier process left.
        self._earlier: dict[str, tuple[float, str | None]] = {}
        self._earlier_states: Counter = Counter()
        if disk is not None:
            for kind in ("keys", "ram_only"):
                for key, value in disk[kind].items():
                    self._put(kind, key, value)
        for changes in overlays:
            self.apply(changes)

    def apply(self, changes: _Changes) -> None:
        """As `_Changes.apply` does to a whole record."""
        for key, value in changes.keys.items():
            self.put("keys", key, value)
        for key, value in changes.ram_only.items():
            self.put("ram_only", key, value)

    def put(self, kind: str, key: str, value: Any) -> None:
        """*key* noted under *kind*: out of the other kind, last in its own."""
        self._drop("ram_only" if kind == "keys" else "keys", key)
        self._drop(kind, key)
        self._put(kind, key, value)

    def _entries(self, kind: str) -> dict[str, Any]:
        return self.keys if kind == "keys" else self.ram_only

    def _put(self, kind: str, key: str, value: Any) -> None:
        self._entries(kind)[key] = value
        parts = _key_parts(key)
        state = parts[1] if len(parts) == 4 else None
        if state is not None:
            self._by_args[kind].setdefault((parts[2], parts[3]), {})[key] = state
        at = _earlier_time(value)
        if at is not None:
            self._earlier[key] = (at, state if key.count(":") >= 3 else None)
            if state is not None:
                self._earlier_states[state] += 1

    def _drop(self, kind: str, key: str) -> None:
        entries = self._entries(kind)
        if key not in entries:
            return
        del entries[key]
        parts = _key_parts(key)
        if len(parts) == 4:
            index = self._by_args[kind]
            same = index.get((parts[2], parts[3]))
            if same is not None:
                same.pop(key, None)
                if not same:
                    del index[(parts[2], parts[3])]
        earlier = self._earlier.pop(key, None)
        if earlier is not None and earlier[1] is not None:
            self._earlier_states[earlier[1]] -= 1
            if self._earlier_states[earlier[1]] <= 0:
                del self._earlier_states[earlier[1]]

    def facts(self, cache_key: str) -> MissFacts:
        """`facts_from_doc` of the record this view stands for."""
        new = _key_parts(cache_key)
        same = None
        earlier_states: frozenset = frozenset()
        newest = None
        if len(new) == 4:
            for kind in ("ram_only", "keys"):
                for key, state in reversed(self._by_args[kind].get((new[2], new[3]), {}).items()):
                    if state != new[1]:
                        same = key
                        break
                if same is not None:
                    break
            earlier_states = frozenset(self._earlier_states)
            if earlier_states and new[1] not in earlier_states:
                newest = max(self._earlier, key=lambda key: self._earlier[key][0])
        last_other = next((key for key in reversed(self.keys) if key != cache_key), None)
        return MissFacts(
            self.keys.get(cache_key), self.ram_only.get(cache_key), same, earlier_states, newest, last_other
        )


# Every live record, for `_reset_after_fork_in_child`.
_LIVE_RECORDS: weakref.WeakSet = weakref.WeakSet()


def _reset_after_fork_in_child() -> None:
    """New locks for every live `StoredKeyRecord` in a forked child (`_after_fork_in_child`)."""
    for record in list(_LIVE_RECORDS):
        record._after_fork_in_child()


if hasattr(os, "register_at_fork"):  # not on Windows, which cannot fork
    os.register_at_fork(after_in_child=_reset_after_fork_in_child)


class StoredKeyRecord:
    """Reads and writes the per-function records of one `Cash` instance."""

    #: Keys kept per function and kind, most recent last.
    KEYS_MAX = 64
    #: States whose ledger a record keeps, most recent last.
    STATES_MAX = 8
    WARNED_MAX = 32
    #: Seconds between two writes of one record. A miss of a cheap function
    #: rewrote the whole file each time (a read, a JSON dump, a rename: more
    #: than the call); what is not written yet is still in memory, where
    #: `read` answers from, and `flush` writes it at shutdown.
    WRITE_INTERVAL = 1.0

    def __init__(self, local_dir: Callable[[], str | None]) -> None:
        """*local_dir* returns the built backend's local directory, or None
        when there is none (no record then)."""
        self._local_dir = local_dir
        # Guards the state below; held for no file access, so a store never
        # waits for a write.
        self._lock = threading.Lock()
        # Serialises this process's rewrites of a record.
        self._io_lock = threading.Lock()
        self._memo: LruMemo[str, tuple[tuple[int, int], Doc]] = LruMemo(RECORDS)
        # path -> when its memo was last found to match the file (monotonic).
        self._checked: dict[str, float] = {}
        # path -> its `_View`, kept while it is in use (`miss_facts`).
        self._views: dict[str, _View] = {}
        self._pending: dict[str, _Changes] = {}
        # Taken by a write in progress; still part of what `read` answers.
        self._writing: dict[str, _Changes] = {}
        self._scheduled: set[str] = set()
        # path -> when its last write ended (monotonic), and the timer that
        # schedules a write held back by `WRITE_INTERVAL`.
        self._last_write: dict[str, float] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._writes: PendingWrites | None = None
        self._closed = False
        _LIVE_RECORDS.add(self)

    def _after_fork_in_child(self) -> None:
        """Start the record's writing afresh in a forked child.

        Only the forking thread survives a fork. The background writer, busy
        rewriting a record at that moment, held ``_io_lock`` -- and in the
        child nothing would ever release it: the child's first record write
        blocked, and its exit waited on that write forever. The write queue
        itself is reset by ``_writes``; the writes the parent had scheduled or
        started are the parent's to finish, so the child forgets them and
        schedules its own changes anew.
        """
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._scheduled = set()
        self._writing = {}
        self._timers = {}  # the parent's timer threads did not survive the fork
        self._views = {}  # built with the parent's writes in progress

    def path(self, func_name: str) -> str | None:
        """Where *func_name*'s record lives, or None.

        In the directory of the backend actually built -- never a configured
        path, so reading a miss reason cannot create a cache directory.
        """
        cache_dir = self._local_dir()
        if not cache_dir:
            return None
        name = hashlib.sha256(func_name.encode("utf-8")).hexdigest()[:32]
        return os.path.join(cache_dir, ".keys", f"{name}.json")

    # -- reading ---------------------------------------------------------

    def read(self, func_name: str) -> Doc:
        """What earlier runs and this one recorded for *func_name* (a copy)."""
        path = self.path(func_name)
        if path is None:
            return _empty()
        # Taken before the file is read: a write that lands in between is
        # then applied twice, which changes nothing, rather than not at all.
        with self._lock:
            overlays = [c.copy() for c in (self._writing.get(path), self._pending.get(path)) if c is not None]
        # Another process's write shows within `WRITE_INTERVAL`: this answers
        # "why did that miss?", and a stat per miss cost more than the rest.
        doc = self._read_disk(path, recheck_after=self.WRITE_INTERVAL)
        for changes in overlays:
            changes.apply(doc)
        return doc

    def miss_facts(self, func_name: str, cache_key: str) -> MissFacts:
        """`facts_from_doc` of what `read` returns for *func_name*, from a view
        kept up to date as keys are noted: a lookup per question, not a copy
        and a walk of the whole record per miss."""
        path = self.path(func_name)
        if path is None:
            return facts_from_doc(_empty(), cache_key)
        disk_doc = self._disk_doc(path, self.WRITE_INTERVAL)  # the file, as `read` sees it
        with self._lock:
            view = self._views.get(path)
            if view is None or view.disk is not disk_doc:
                overlays = [c for c in (self._writing.get(path), self._pending.get(path)) if c is not None]
                view = self._views[path] = _View(disk_doc, overlays)
            return view.facts(cache_key)

    def _read_disk(self, path: str, *, recheck_after: float = 0.0) -> Doc:
        """The record on disk, through a memo on its (mtime, size) (a copy).
        A memo checked against the file less than *recheck_after* seconds
        ago is taken without a stat."""
        doc = self._disk_doc(path, recheck_after)
        return _empty() if doc is None else {kind: dict(value) for kind, value in doc.items()}

    def _disk_doc(self, path: str, recheck_after: float) -> Doc | None:
        """`_read_disk`'s answer as the memo's own record, never changed once
        kept (callers must not change it either); None for no file."""
        with self._lock:
            memo = self._memo.get(path)
            checked = self._checked.get(path)
        now = time.monotonic()
        if memo is not None and checked is not None and now - checked < recheck_after:
            return memo[1]
        try:
            st = os.stat(path)
        except OSError:
            return None
        if memo is not None and memo[0] == (st.st_mtime_ns, st.st_size):
            with self._lock:
                self._checked[path] = now
            return memo[1]
        try:
            # A nested call reads this while the outer call's file tracker is
            # live, and it must not become that entry's dependency.
            with untracked(), open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            # Another process mid-rewrite (on Windows a read that overlaps the
            # rename fails): the last record read beats none.
            return memo[1] if memo is not None else None
        return self._remember(path, st, data)

    def _remember(self, path: str, st: os.stat_result, data: Any) -> Doc:
        doc = _empty()
        if isinstance(data, dict):
            for kind in _KINDS:
                value = data.get(kind)
                if isinstance(value, dict):
                    doc[kind] = value
        with self._lock:
            self._memo[path] = ((st.st_mtime_ns, st.st_size), doc)
            self._checked[path] = time.monotonic()
        return doc

    # -- noting ----------------------------------------------------------

    def note_stored(
        self, func_name: str, cache_key: str, ttl: int | None, flat_ledger: Callable[[str], dict | None]
    ) -> None:
        """*cache_key* reached disk. Written soon, in the background.

        *flat_ledger* gives a state's ledger, for the next run's "what changed".
        """
        path = self._note(func_name, cache_key, "keys", [time.time(), ttl], flat_ledger)
        if path is not None:
            self._schedule(path)

    def note_ram_only(
        self, func_name: str, cache_key: str, why: str, flat_ledger: Callable[[str], dict | None]
    ) -> None:
        """A result kept in RAM only, so the next run does not call it "new
        arguments". Written with the record's next write, or at `flush`: a
        result kept in RAM is a quick one, and a write per miss would cost
        more than the body."""
        self._note(func_name, cache_key, "ram_only", [time.time(), why], flat_ledger)

    def note_warning_shown(self, func_name: str, digest: str) -> None:
        """A warning was shown. Written with the record's next write, never
        now: this runs before the first call's lookup, and creating the cache
        directory here would reorder the stamps the clear check reads."""
        path = self.path(func_name)
        if path is None:
            return
        with self._lock:
            changes = self._changes(path, func_name)
            changes.warned[digest] = time.time()
            _trim(changes.warned, self.WARNED_MAX)

    def _changes(self, path: str, func_name: str) -> _Changes:
        changes = self._pending.get(path)
        if changes is None:
            changes = self._pending[path] = _Changes(func_name)
        return changes

    def _note(
        self, func_name: str, cache_key: str, kind: str, value: list, flat_ledger: Callable[[str], dict | None]
    ) -> str | None:
        path = self.path(func_name)
        if path is None:
            return None
        parts = cache_key.rsplit(":", 3)
        state = parts[1] if len(parts) == 4 else None
        with self._lock:
            changes = self._changes(path, func_name)
            other = changes.ram_only if kind == "keys" else changes.keys
            other.pop(cache_key, None)
            mine = getattr(changes, kind)
            _put_last(mine, cache_key, value)
            trimmed = len(mine) > self.KEYS_MAX
            _trim(mine, self.KEYS_MAX)
            view = self._views.get(path)
            if view is not None:
                if trimmed:  # an older note left the record: built again on the next miss
                    del self._views[path]
                else:
                    view.put(kind, cache_key, value)
            if state in changes.states:
                _put_last(changes.states, state, changes.states[state])
            elif state is not None:
                memo = self._memo.get(path)
                on_disk = memo is not None and state in memo[1]["states"]
                changes.states[state] = None if on_disk else flat_ledger(state)
                _trim(changes.states, self.STATES_MAX)
        return path

    # -- writing ---------------------------------------------------------

    def _schedule(self, path: str, *, hold: bool = False) -> None:
        """Have *path* written: now, or once `WRITE_INTERVAL` has passed since
        its last write. *hold* always goes through the timer: the writer task
        itself must not submit to its own queue."""
        with self._lock:
            if path in self._scheduled or path in self._timers:
                return  # the queued (or held back) write takes this change too
            wait = self._last_write.get(path, float("-inf")) + self.WRITE_INTERVAL - time.monotonic()
            if (wait > 0 or hold) and not self._closed:
                wait = max(wait, 0.0)
                # Written a moment ago: hold this one back. A timer, not a
                # sleep in the writer, so nothing that waits for the write
                # queue waits for the interval.
                timer = threading.Timer(wait, self._release_held, (path,))
                timer.daemon = True
                self._timers[path] = timer
                timer.start()
                return
            if self._closed:
                queue = None
            else:
                if self._writes is None:
                    self._writes = PendingWrites(max_workers=1)
                    # A drain of the write queues (after a notebook cell, or
                    # before a listing) also writes what is held back.
                    self._writes.on_drain(self._submit_held)
                queue = self._writes
                self._scheduled.add(path)
        if queue is None:
            self._write_now(path)
            return
        try:
            queue.submit(f"stored-keys:{path}", self._write_while_pending, path)
        except RuntimeError:  # shut down meanwhile
            with self._lock:
                self._scheduled.discard(path)
            self._write_now(path)

    def _release_held(self, path: str) -> None:
        """Timer callback: schedule the write `_schedule` held back."""
        with self._lock:
            if self._timers.pop(path, None) is None:
                return  # flushed meanwhile
        self._schedule(path)

    def _submit_held(self) -> None:
        """Drain hook: submit every write held back by `WRITE_INTERVAL` now."""
        with self._lock:
            timers, self._timers = self._timers, {}
            queue = self._writes
        for path, timer in timers.items():
            timer.cancel()
            with self._lock:
                if path in self._scheduled:
                    continue  # a queued write takes it
                self._scheduled.add(path)
            try:
                if queue is None:
                    raise RuntimeError("closed")
                queue.submit(f"stored-keys:{path}", self._write_while_pending, path)
            except RuntimeError:
                with self._lock:
                    self._scheduled.discard(path)
                self._write_now(path)

    def _write_while_pending(self, path: str) -> None:
        """Writer task: write *path*.

        Gives up the slot and checks for a late change in one step, so a
        change noted meanwhile is held for the next write, after the interval
        (a task must not submit to its own queue: it would wait on itself).
        """
        self._write_now(path)
        with self._lock:
            self._scheduled.discard(path)
            late = path in self._pending and not self._closed
        if late:
            self._schedule(path, hold=True)

    def _write_now(self, path: str) -> bool:
        """Write the changes pending for *path*. False when there were none.
        Never raises: the record is a diagnostic aid."""
        with self._lock:
            changes = self._pending.pop(path, None)
            if changes is None:
                return False
            self._writing[path] = changes
        try:
            with self._io_lock:
                doc = self._read_disk(path)
                changes.apply(doc)
                for kind, most in (
                    ("keys", self.KEYS_MAX),
                    ("ram_only", self.KEYS_MAX),
                    ("states", self.STATES_MAX),
                    ("warned", self.WARNED_MAX),
                ):
                    _trim(doc[kind], most)
                keys_dir = os.path.dirname(path)
                recreate_cache_dir(os.path.dirname(keys_dir))
                os.makedirs(keys_dir, exist_ok=True)
                tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
                with untracked():
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump({"func": changes.func, **doc}, fh)
                    replace_with_retry(tmp, path)
                    # What this process just wrote is what its next miss reads.
                    self._remember(path, os.stat(path), doc)
        except (OSError, TypeError, ValueError):
            logger.debug("could not write the stored-key record %s", path, exc_info=True)
        finally:
            with self._lock:
                self._writing.pop(path, None)
                self._last_write[path] = time.monotonic()
        return True

    def flush(self) -> None:
        """Wait for queued writes, then write every change still pending. Never raises."""
        with self._lock:
            timers, self._timers = self._timers, {}
        for timer in timers.values():
            timer.cancel()  # their changes are pending: written below
        queue = self._writes
        if queue is not None and not queue.in_worker_thread():
            queue.wait_all()
        with self._lock:
            paths = list(self._pending)
        for path in paths:
            self._write_now(path)

    def close(self) -> None:
        """`flush`, and stop the writer (bounded, as it runs at exit)."""
        with self._lock:
            self._closed = True
            queue, self._writes = self._writes, None
        if queue is not None:
            queue.shutdown(wait=True)
        self.flush()
