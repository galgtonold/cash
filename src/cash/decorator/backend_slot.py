"""The backend a `Cash` stores in, built from its config on first use."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from ..backends.factory import build_backend_from_config

if TYPE_CHECKING:
    from ..backends import CacheBackend
    from ..config import CashConfig


class BackendSlot:
    """Holds the one backend of a `Cash`: the one it was given, or one built
    from its config the first time the cache is used.

    Building lazily keeps ``Cash()`` free of filesystem I/O, threads and
    directory scans; a script that decorates functions and never calls them
    pays for none of it.
    """

    def __init__(self, config: CashConfig, given: CacheBackend | None = None) -> None:
        self._config = config
        self._backend = given
        self._lock = threading.Lock()

    @property
    def backend(self) -> CacheBackend:
        """The backend, built from the config on first access."""
        backend = self._backend
        if backend is not None:
            return backend
        with self._lock:
            if self._backend is None:
                self._backend = build_backend_from_config(self._config)
            return self._backend

    @backend.setter
    def backend(self, value: CacheBackend) -> None:
        self._backend = value

    @property
    def built(self) -> CacheBackend | None:
        """The backend if one has been built or given, else ``None``; never builds one."""
        return self._backend

    def local_dir(self) -> str | None:
        """The built backend's local directory, or ``None``; never builds one."""
        backend = self._backend
        return backend.local_dir if backend is not None else None

    def tier_default_ttl(self) -> int | None:
        """The ``default_ttl`` of the first tier that has one, as configured now."""
        backend = self._backend
        return backend.default_ttl if backend is not None else None

    def entry_ttl(self, ttl: int | None, metadata: Any) -> int | None:
        """The ttl a stored entry is judged by.

        The decorator's ``ttl=`` when it has one -- a per-function setting,
        applied as it stands now, in both directions. Otherwise the SHORTER of
        the ttl the entry was written with and the tier's ``default_ttl`` as
        configured now: lowering a tier's default from a day to 5 seconds left
        every entry written under the day being served,
        while lowering a decorator's ttl took effect at once.
        """
        if ttl is not None:
            return ttl
        found = [t for t in (getattr(metadata, "ttl", None), self.tier_default_ttl()) if t is not None]
        return min(found) if found else None
