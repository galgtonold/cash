"""``__cash_key__``: a class says what identifies its instances in a key.

A class holding eight large frames keyed by content costs a full read of
them on every call, and a method that returns their row count pays it too.
``__cash_key__`` lets the class name what identifies an instance, a
version, a path, a content id, and the key uses that instead of the data.

The risk it brings is a key that stays put while the data changes. So the
first time each object is keyed this way in a process, `KeyCheck` reads
its content on a background thread and compares it with what the same key
held before, in this process or an earlier one (a small record beside the
cache). Two different contents behind one key warn KEY-STALE-CASH-KEY once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import queue
import threading
import weakref
from collections.abc import Callable
from typing import Any

from ..backends.cache_dir import CASH_KEYS_FILENAME, create_temp_file
from ..canonical_form import canonical_bytes, object_state
from ..content_hashers import BUILTIN_CONTENT
from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning
from ..tracking.tracker_context import untracked

logger = logging.getLogger(__name__)

__all__ = ["CASH_KEY", "KeyCheck", "cash_key_method", "content_digest"]

#: The method a class defines to name what identifies its instances.
CASH_KEY = "__cash_key__"


def cash_key_method(value: Any) -> Callable[[Any], Any] | None:
    """The ``__cash_key__`` of *value*'s class, or ``None``.

    Looked up on the class, as Python looks up its own dunders: a class passed
    as a value is not keyed by the method it defines for its instances, and
    ``__cash_key__ = None`` on a subclass turns an inherited one off.
    """
    t = type(value)
    try:
        return _METHODS[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _lookup(t)
    if len(_METHODS) >= _METHODS_MAX:
        _METHODS.clear()
    fn = _METHODS[t] = _lookup(t)
    return fn


def cash_key_method_of_type(t: type) -> Callable[[Any], Any] | None:
    """The ``__cash_key__`` instances of *t* are keyed by, or ``None``."""
    return _lookup(t)


def _lookup(t: type) -> Callable[[Any], Any] | None:
    if issubclass(t, type):
        return None
    try:
        fn = getattr(t, CASH_KEY, None)
    except Exception:  # noqa: BLE001 - an exotic metaclass: no key method
        return None
    return fn if callable(fn) else None


#: ``class -> its __cash_key__ or None``, asked of every value a key walk
#: meets. A class's methods are fixed once it is defined; a redefined class
#: is a new class and a new entry.
_METHODS: dict[type, Callable[[Any], Any] | None] = {}
_METHODS_MAX = 4096


def type_name(value: Any) -> str:
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def content_digest(value: Any) -> str | None:
    """A digest of what *value* holds, as the key would see it without
    ``__cash_key__``, or ``None`` when some part of it cannot be read.

    Each attribute is taken on its own, so frames, arrays and tables go
    through their content hashers and one attribute that cannot be pickled
    (a lock, a connection) is left out rather than hiding the rest.
    """
    parts = []
    try:
        state = object_state(value)
    except Exception:  # noqa: BLE001 - nothing to compare
        return None
    if not state:
        return None
    for name in sorted(state):
        try:
            payload = canonical_bytes(state[name], BUILTIN_CONTENT)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError, RecursionError):
            continue
        parts.append(f"{name}:{hashlib.sha256(payload).hexdigest()}")
    if not parts:
        return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class _Record:
    """``key id -> content digest``, beside the cache, shared by processes.

    Advisory, like the eviction notes: a lost or unreadable file only means
    the next check has nothing to compare with. Without a cache directory
    (a RAM-only cache) the record lives for the process.
    """

    #: Key ids kept; the oldest go first.
    MAX_IDS = 10_000

    def __init__(self, local_dir: Callable[[], str | None]) -> None:
        self._local_dir = local_dir
        self._memory: dict[str, str] = {}

    def _path(self) -> str | None:
        try:
            directory = self._local_dir()
        except Exception:  # noqa: BLE001 - no backend yet: keep it in memory
            return None
        return os.path.join(directory, CASH_KEYS_FILENAME) if directory else None

    def _load(self, path: str) -> dict[str, str]:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def seen(self, key_id: str, digest: str) -> str | None:
        """The digest recorded for *key_id*; records *digest* when there is none."""
        path = self._path()
        if path is None:
            return self._memory.setdefault(key_id, digest)
        with untracked():
            data = self._load(path)
            known = data.get(key_id)
            if isinstance(known, str):
                return known
            data[key_id] = digest
            while len(data) > self.MAX_IDS:
                data.pop(next(iter(data)))
            try:
                fd, tmp = create_temp_file(os.path.dirname(path))
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp, path)
            except OSError:
                logger.debug("could not write %s", path, exc_info=True)
        return digest


class KeyCheck:
    """Checks, once per object per process, that a ``__cash_key__`` still
    names the same content it named before.

    The reading happens on one background thread, so a call keyed by
    ``__cash_key__`` never waits for the content it was written to skip.
    """

    def __init__(self, local_dir: Callable[[], str | None], enabled: Callable[[], bool]) -> None:
        self._record = _Record(local_dir)
        self._enabled = enabled
        self._lock = threading.Lock()
        #: id(obj) -> weakref of objects already queued in this process.
        self._queued: dict[int, Any] = {}
        #: Key ids already warned about in this process.
        self._warned: set[str] = set()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        # A child forked while the thread held the lock, or had checks
        # queued, would inherit a held lock and a queue nothing drains.
        if hasattr(os, "register_at_fork"):
            me = weakref.ref(self)
            os.register_at_fork(after_in_child=lambda: (c := me()) is not None and c._after_fork())

    def _after_fork(self) -> None:
        self._lock = threading.Lock()
        self._queued = {}
        self._queue = queue.Queue()
        self._thread = None

    def note(self, value: Any, key_id: str, method: Callable) -> None:
        """Queue *value* for a check, unless it was queued before."""
        try:
            if not self._enabled():
                return
        except Exception:  # noqa: BLE001 - a config read must never break a call
            return
        with self._lock:
            known = self._queued.get(id(value))
            if known is not None and known() is value:
                return
            try:
                ref = weakref.ref(value, lambda _r, key=id(value): self._queued.pop(key, None))
            except TypeError:
                return  # no weak references: it cannot be checked without keeping it alive
            self._queued[id(value)] = ref
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="cash-key-check", daemon=True)
                self._thread.start()
        self._queue.put((ref, key_id, method))

    def wait(self) -> None:
        """Block until every queued check has run (for tests)."""
        self._queue.join()

    def _run(self) -> None:
        while True:
            ref, key_id, method = self._queue.get()
            try:
                value = ref()
                if value is not None:
                    self._check(value, key_id, method)
            except Exception:  # a check must never surface as an error
                logger.debug("__cash_key__ check failed", exc_info=True)
            finally:
                value = None
                self._queue.task_done()

    def _check(self, value: Any, key_id: str, method: Callable) -> None:
        digest = content_digest(value)
        if digest is None:
            return
        recorded = self._record.seen(key_id, digest)
        if recorded == digest or key_id in self._warned:
            return
        # Read again: an object being filled in while it was read is not a
        # stale key, and only a content that holds still is reported.
        if content_digest(value) != digest:
            return
        self._warned.add(key_id)
        code = getattr(method, "__code__", None)
        location = (code.co_filename, code.co_firstlineno) if code is not None else None
        name = type(value).__qualname__
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "KEY-STALE-CASH-KEY",
            f"{name}.__cash_key__() returned the same key for {name} values holding different data, "
            f"so cached results computed from one are served for the other.",
            f"make {name}.__cash_key__() return something that changes whenever the data does "
            f"(a version, a content id, the source file's modification time), then clear the "
            f"results stored under the old key with cache_clear() on the functions that took it.",
            location=location,
        )
