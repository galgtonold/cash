"""A small ``{key: value}`` map kept as a versioned JSON file in the cache dir.

Shared by the loop-split verdicts (:mod:`.loop_split`), the measured
compute costs (:mod:`.compute_baselines`) and the miss guard's verdicts
(:mod:`.statement.miss_guard`). All follow the same rules, so they live here
once:

* **Lazy.** The file is read on first use, once per session.
* **Best-effort.** A missing, unreadable, corrupt or future-versioned file
  leaves the store empty; every entry is validated as it is loaded and a bad
  one is dropped. An empty store must always mean "the behaviour without it".
* **Atomic.** A write goes to a pid-suffixed temporary file and is moved into
  place with :func:`~cash._paths.replace_with_retry`, never a bare
  ``os.replace``: on Windows that is DENIED, not delayed, while any handle has
  the destination open, and the failure would be swallowed at debug level --
  the entry gone from disk while it stays in memory.
* **One per directory.** :class:`StoreRegistry` hands every caller the same
  instance for a cache dir. For the loop split that is a correctness
  requirement, not an optimisation: the store loads once per session, so two
  instances diverge the moment one of them records.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from cash._paths import replace_with_retry
from cash.backends.cache_dir import recreate_cache_dir

if TYPE_CHECKING:
    from cash.backends import CacheBackend

logger = logging.getLogger(__name__)

V = TypeVar("V")


def resolve_cache_dir(backend: CacheBackend | None) -> str | None:
    """The on-disk cache directory behind *backend* (``local_dir``), or None.

    None means there is nowhere to persist — a pure in-memory backend, which
    has no restart to survive anyway, so a store there is session-scoped.
    """
    return (backend.local_dir or None) if backend is not None else None


class VersionedJsonStore(Generic[V]):
    """``{str: V}`` persisted as ``{"version": VERSION, FIELD: {...}}``.

    A subclass names its file, version and field, and says which loaded values
    it accepts (:meth:`_load_value`) and how a value is written
    (:meth:`_dump_value`).
    """

    FILENAME: ClassVar[str]
    VERSION: ClassVar[int]
    FIELD: ClassVar[str]
    LOG_TAG: ClassVar[str]

    def __init__(self, cache_dir: str | None) -> None:
        self._path = os.path.join(cache_dir, self.FILENAME) if cache_dir else None
        self._items: dict[str, V] = {}
        self._loaded = False

    def _load_value(self, value: Any) -> V | None:
        """*value* as read from the file, or ``None`` to drop the entry."""
        raise NotImplementedError

    def _dump_value(self, value: V) -> Any:
        return value

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            logger.debug("[%s] no readable store at %s", self.LOG_TAG, self._path)
            return
        if not isinstance(doc, dict) or doc.get("version") != self.VERSION:
            return
        items = doc.get(self.FIELD)
        if not isinstance(items, dict):
            return
        for key, raw in items.items():
            value = self._load_value(raw) if isinstance(key, str) else None
            if value is not None:
                self._items[key] = value

    def _write(self) -> bool:
        """Write the store out; False when it could not be (it stays in memory)."""
        if not self._path:
            return False
        doc = {"version": self.VERSION, self.FIELD: {k: self._dump_value(v) for k, v in sorted(self._items.items())}}
        tmp_path = f"{self._path}.{os.getpid()}.tmp"
        try:
            recreate_cache_dir(os.path.dirname(self._path))
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            replace_with_retry(tmp_path, self._path)
            return True
        except OSError:
            logger.debug("[%s] could not persist to %s", self.LOG_TAG, self._path)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            return False

    def _delete_file(self) -> None:
        if self._path:
            with contextlib.suppress(OSError):
                os.unlink(self._path)


S = TypeVar("S", bound=VersionedJsonStore)


class StoreRegistry(Generic[S]):
    """One store per cache dir, process-wide."""

    def __init__(self, make: Callable[[str | None], S], on_create: Callable[[S], None] | None = None) -> None:
        self._make = make
        self._on_create = on_create
        self._stores: dict[str | None, S] = {}

    def get(self, cache_dir: str | None) -> S:
        """The shared store for *cache_dir*."""
        store = self._stores.get(cache_dir)
        if store is None:
            store = self._stores[cache_dir] = self._make(cache_dir)
            if self._on_create is not None:
                self._on_create(store)
        return store

    def for_backend(self, backend: Any) -> S | None:
        """The shared store for *backend*'s cache dir, or ``None`` if it cannot
        be resolved -- an unresolvable store means "no store"."""
        try:
            return self.get(resolve_cache_dir(backend))
        except Exception:  # every store here is an optimisation or a nicety
            logger.debug("could not resolve a store", exc_info=True)
            return None
