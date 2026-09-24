"""The record of which keys earlier runs stored, kept beside the cache so a
new process can say why its first call missed.

One small JSON file per function under ``<cache>/.keys/``. Changes are kept
in memory and written by a background writer, several stores to one file at a
time; results kept in RAM only and warnings shown ride along with the next
write, or with `StoredKeyRecord.flush` at shutdown.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .._paths import replace_with_retry
from ..backends._writes import PendingWrites
from ..backends.cache_dir import recreate_cache_dir
from ..tracking.tracker_context import untracked

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


class StoredKeyRecord:
    """Reads and writes the per-function records of one `Cash` instance."""

    #: Keys kept per function and kind, most recent last.
    KEYS_MAX = 64
    #: States whose ledger a record keeps, most recent last.
    STATES_MAX = 8
    WARNED_MAX = 32
    _MEMO_MAX = 256

    def __init__(self, local_dir: Callable[[], str | None]) -> None:
        """*local_dir* returns the built backend's local directory, or None
        when there is none (no record then)."""
        self._local_dir = local_dir
        # Guards the state below; held for no file access, so a store never
        # waits for a write.
        self._lock = threading.Lock()
        # Serialises this process's rewrites of a record.
        self._io_lock = threading.Lock()
        self._memo: dict[str, tuple[tuple[int, int], Doc]] = {}
        self._pending: dict[str, _Changes] = {}
        # Taken by a write in progress; still part of what `read` answers.
        self._writing: dict[str, _Changes] = {}
        self._scheduled: set[str] = set()
        self._writes: PendingWrites | None = None
        self._closed = False

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
        doc = self._read_disk(path)
        for changes in overlays:
            changes.apply(doc)
        return doc

    def _read_disk(self, path: str) -> Doc:
        """The record on disk, through a memo on its (mtime, size)."""
        with self._lock:
            memo = self._memo.get(path)
        try:
            st = os.stat(path)
        except OSError:
            return _empty()
        if memo is not None and memo[0] == (st.st_mtime_ns, st.st_size):
            return {kind: dict(value) for kind, value in memo[1].items()}
        try:
            # A nested call reads this while the outer call's file tracker is
            # live, and it must not become that entry's dependency.
            with untracked(), open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            # Another process mid-rewrite (on Windows a read that overlaps the
            # rename fails): the last record read beats none.
            if memo is not None:
                return {kind: dict(value) for kind, value in memo[1].items()}
            return _empty()
        return self._remember(path, st, data)

    def _remember(self, path: str, st: os.stat_result, data: Any) -> Doc:
        doc = _empty()
        if isinstance(data, dict):
            for kind in _KINDS:
                value = data.get(kind)
                if isinstance(value, dict):
                    doc[kind] = value
        with self._lock:
            if len(self._memo) >= self._MEMO_MAX:
                self._memo.clear()
            self._memo[path] = ((st.st_mtime_ns, st.st_size), doc)
        return {kind: dict(value) for kind, value in doc.items()}

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
            _trim(mine, self.KEYS_MAX)
            if state in changes.states:
                _put_last(changes.states, state, changes.states[state])
            elif state is not None:
                memo = self._memo.get(path)
                on_disk = memo is not None and state in memo[1]["states"]
                changes.states[state] = None if on_disk else flat_ledger(state)
                _trim(changes.states, self.STATES_MAX)
        return path

    # -- writing ---------------------------------------------------------

    def _schedule(self, path: str) -> None:
        with self._lock:
            if path in self._scheduled:
                return  # the queued write takes this change too
            if self._closed:
                queue = None
            else:
                if self._writes is None:
                    self._writes = PendingWrites(max_workers=1)
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

    def _write_while_pending(self, path: str) -> None:
        """Writer task: write *path* until no change for it is left.

        Checks for a late change and gives up the slot in one step, so a
        change noted meanwhile either is written here or schedules a new task
        (a task must not submit to its own queue: it would wait on itself).
        """
        while True:
            while self._write_now(path):
                pass
            with self._lock:
                if path not in self._pending or self._closed:
                    self._scheduled.discard(path)
                    return

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
        return True

    def flush(self) -> None:
        """Wait for queued writes, then write every change still pending. Never raises."""
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
