"""Tiered (multi-level) cache backend for Cash."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ._base import CacheBackend, MetadataDict
from .cascading_backend import _MultiBackendMixin
from .serialization import Serializer

logger = logging.getLogger(__name__)

__all__ = ["TieredBackend"]

class TieredBackend(_MultiBackendMixin, CacheBackend):
    """
    Backend that manages multiple cache tiers (e.g., Memory -> File -> S3).
    Implements smart promotion and read-repair.
    """

    def __init__(self, backends: list[CacheBackend], promotion_policy: Callable[[float, int], bool] | None = None) -> None:
        """
        Args:
            backends: List of cache backends, ordered by speed (fastest first).
            promotion_policy: Callable taking (execution_time, size_bytes) and returning True if should promote.
        """
        self.backends = backends
        self.promotion_policy = promotion_policy or self._default_promotion_policy
        self._disk_bandwidth_est = 100 * 1024 * 1024 # 100 MB/s conservative estimate

    def _default_promotion_policy(self, execution_time: float, size_bytes: int) -> bool:
        """
        Default policy: Promote if execution time is significant AND
        recomputing is likely slower than reading from next tier.
        """
        # Threshold 1: Logic must take at least 1.0s to be worth caching on disk
        if execution_time < 1.0:
            return False

        # Threshold 2: Reading from disk shouldn't be slower than recomputing
        # Time to read = Size / Bandwidth
        read_time = size_bytes / self._disk_bandwidth_est

        # If execution time is much larger than read time, cache it.
        return execution_time > read_time

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        for i, backend in enumerate(self.backends):
            metadata, value = backend.get(key)
            # Key-presence test: metadata is None when the child backend
            # reports "key absent" (per its API contract). A non-None
            # metadata dict with a None value means the user genuinely
            # cached None — still a hit.
            if metadata is not None:
                # Read-Repair / Promotion to faster tiers
                # If found in Tier 2 (File), promote to Tier 1 (Memory)
                for j in range(i):
                    # Always promote to faster tiers on read?
                    # Generally yes, L1 (Memory) should hold hot items.
                    # Exception: if it's too huge for memory?
                    # MemoryBackend handles its own eviction, so we can just try setting it.
                    try:
                        self.backends[j].set(key, value, metadata)
                    except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
                        logger.warning(
                            "Failed to promote key '%s' to tier %d (%s): %s",
                            key, j, type(self.backends[j]).__name__, e,
                        )

                # Inject source information
                metadata['source'] = getattr(type(backend), 'source_label', None) or type(backend).__name__

                return metadata, value
        return None, None

    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None) -> None:
        if not self.backends:
            return

        # Keep a reference to the original dict so we can propagate storage info back
        original_metadata = metadata
        metadata = dict(metadata) if metadata is not None else {}
        stored_destinations = []

        # Always write to Tier 0 (Memory)
        try:
            self.backends[0].set(key, value, metadata, serializer)
            stored_destinations.append("RAM")
        except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
            logger.warning(
                "Failed to write key '%s' to tier 0 (%s): %s",
                key, type(self.backends[0]).__name__, e,
            )

        # Check promotion for subsequent tiers. The promotion decision is
        # made per-tier so a single set() can land in some tiers and skip
        # others — e.g. a 20 MB DataFrame goes to RAM + DISK but skips
        # Redis (10 MB cap).
        if len(self.backends) > 1:
            exec_time = metadata.get('execution_time', 0)
            size = metadata.get('size', 0)

            # Check if force_persist is set via @cash:persist annotation
            force_persist = metadata.get('force_persist', False)

            # Universal compute floor — same for every tier past tier 0.
            past_compute_floor = force_persist or self.promotion_policy(exec_time, size)

            for i in range(1, len(self.backends)):
                backend = self.backends[i]
                if not past_compute_floor:
                    continue
                cap = getattr(type(backend), 'max_size_bytes', None)
                if cap is not None and size and size > cap:
                    # This tier doesn't want objects this large.
                    logger.debug(
                        "[TIERED] Skipping %s for key %r: size %d > cap %d",
                        type(backend).__name__, key, size, cap,
                    )
                    continue
                try:
                    backend.set(key, value, metadata, serializer)
                    _label = getattr(type(backend), 'source_label', None) or type(backend).__name__
                    stored_destinations.append(_label)
                except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
                    logger.warning("[TIERED] Failed to write to backend %s: %s", type(backend).__name__, e)

        # Update metadata with storage info so UI can see it immediately
        if metadata is not None:
             metadata['storage'] = stored_destinations

        # Propagate storage info back to the caller's original metadata dict
        if original_metadata is not None:
            original_metadata['storage'] = stored_destinations

        # Log visibility
        if stored_destinations:
            logger.debug("[STORAGE] Stored in: %s", ', '.join(stored_destinations))

