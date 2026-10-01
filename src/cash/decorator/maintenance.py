"""Removing stored entries: one function's (``cache_clear``) and the expired
ones (``Cash.cleanup``)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..backends._base import entry_expired
from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheStoreFailedWarning
from .cache_metadata import CacheMetadata

if TYPE_CHECKING:
    from ..backends import CacheBackend
    from .backend_slot import BackendSlot
    from .registry import FunctionRegistry

logger = logging.getLogger(__name__)


class Maintenance:
    """Deletes entries from one `Cash` instance's backend."""

    def __init__(self, registry: FunctionRegistry, backend_slot: BackendSlot) -> None:
        self._registry = registry
        self._backend_slot = backend_slot

    def delete_function_entries(self, func_name: str) -> None:
        """Delete all backend cache entries whose key starts with *func_name*,
        tell running processes, and say which entries could not be removed.

        Other processes (and other `Cash` instances on the folder) may hold
        the deleted results in their RAM tiers. Moving the disk tier's
        generation is what tells them to drop those, as ``cash clear
        --function`` does; without it they went on serving what was cleared.
        """
        backend = self._backend_slot.backend
        keys = _function_keys(backend, func_name)
        deleted, survivors = _delete_keys(backend, keys)
        if deleted:
            try:
                backend.bump_generation()
            except Exception:  # the local clear is done either way
                logger.debug("Could not tell other processes about the clear", exc_info=True)
        if survivors:
            warn_diagnostic(
                CashCacheStoreFailedWarning,
                "CACHE-CLEAR-INCOMPLETE",
                f"{func_name}.cache_clear() could not remove {len(survivors)} of its "
                f"{len(keys)} cache entr{'y' if len(keys) == 1 else 'ies'}; "
                f"{'it is' if len(survivors) == 1 else 'they are'} still stored and "
                f"will be served.",
                "another process has the entry files open (on Windows: a reader, a "
                "virus scanner, an indexer). Close it and clear again, or run "
                "`cash clear --function` once it has let go.",
            )

    def cleanup(self, max_age: int | None = None) -> int:
        """Remove expired entries, and with *max_age* those older than that
        many seconds whatever their TTL; return how many were removed."""
        backend = self._backend_slot.backend
        now = time.time()
        tier_default = backend.default_ttl

        def is_expired(raw_metadata: Any) -> bool:
            try:
                metadata = CacheMetadata.from_dict(raw_metadata)
                timestamp = metadata.timestamp or 0
                age = now - timestamp

                if max_age is not None and age > max_age:
                    return True

                # The rule a read applies, so cleanup removes exactly what
                # would no longer be served.
                return entry_expired(raw_metadata, tier_default, now, current=self._current_ttl(metadata.func_name))
            except (AttributeError, TypeError, ValueError):
                return True

        return backend.cleanup_expired(is_expired)

    def _current_ttl(self, func_name: str | None) -> float | None:
        """The ``ttl=`` a call of *func_name* would judge its entries by now,
        when that function is decorated in this process."""
        cf = self._registry.cached.get(func_name) if func_name else None
        return self._registry.effective_ttl(func_name, cf.ttl) if cf is not None else None


def _function_keys(backend: CacheBackend, func_name: str) -> list[str]:
    """The stored keys of *func_name*'s entries; none when listing fails."""
    prefix = f"{func_name}:"
    try:
        return [
            k for k in (CacheMetadata.from_dict(e).key or "" for e in backend.list_entries()) if k.startswith(prefix)
        ]
    except (OSError, RuntimeError, KeyError):
        logger.debug("Failed to list cache entries for %s", func_name, exc_info=True)
        return []


def _delete_keys(backend: CacheBackend, keys: list[str]) -> tuple[int, list[str]]:
    """Delete *keys*; return how many deletes ran and the keys still stored."""
    deleted = 0
    survivors: list[str] = []
    for key in keys:
        try:
            backend.delete(key)
            deleted += 1
            # A file another process holds open cannot be removed on
            # Windows; the delete says nothing, and the next call would
            # be served the entry that was meant to be gone.
            if backend.get_metadata(key) is not None:
                survivors.append(key)
        except Exception:  # one entry must not stop the clear
            logger.debug("Failed to clear cache entry %s", key, exc_info=True)
            survivors.append(key)
    return deleted, survivors
