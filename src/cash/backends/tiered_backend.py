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

    # Map a backend implementation class to the cost-model backend kind used
    # to predict restore time. Anything unmapped is treated as disk (the
    # cost model itself also falls back to "disk" for unknown kinds).
    _BACKEND_KIND_BY_CLASS = {
        "InMemoryBackend": "ram",
        "FileBackend": "disk",
        "SQLiteBackend": "disk",
        "RedisBackend": "redis",
        "S3Backend": "s3",
    }

    def __init__(
        self,
        backends: list[CacheBackend],
        promotion_policy: Callable[[float, int], bool] | None = None,
        *,
        min_persist_compute_s: float = 1.0,
        min_persist_savings_pct: float = 0.20,
    ) -> None:
        """
        Args:
            backends: List of cache backends, ordered by speed (fastest first).
            promotion_policy: Callable taking (execution_time, size_bytes) and returning True if should promote.
            min_persist_compute_s: Compute floor for the serialization-aware
                decision — nothing below this is promoted past tier 0. The
                default (1.0 s) matches the fallback ``_default_promotion_policy``;
                the factory lowers it to 0.1 s for the smart-persistence stack.
            min_persist_savings_pct: Required fraction of compute time that a
                cache hit must save to be worth promoting. Mirrors
                ``CashConfig.min_cache_savings_pct`` (Gate A's threshold).
        """
        self.backends = backends
        self.promotion_policy = promotion_policy or self._default_promotion_policy
        self._min_persist_compute_s = min_persist_compute_s
        self._min_persist_savings_pct = min_persist_savings_pct
        # Once-per-session dedup for the oversize-refusal warning.
        self._warned_oversize = False

    def _promotion_backend_kind(self) -> str:
        """Cost-model backend kind of the first tier past RAM (the primary
        persistence target). Used to predict restore cost at ``set`` time."""
        if len(self.backends) > 1:
            name = type(self.backends[1]).__name__
            return self._BACKEND_KIND_BY_CLASS.get(name, "disk")
        return "disk"

    def _cost_model_promote(
        self, type_name: str, size_bytes: int, execution_time: float, backend_kind: str
    ) -> bool:
        """Serialization-aware promotion decision (the same rule Gate A uses):
        promote only when recomputing costs more than the predicted restore.

        ``promote if execution_time - est_restore_time > min_savings * execution_time``

        The prediction comes from the fitted ``cost_model`` (serialize+write /
        read+deserialize end-to-end), so — unlike the old raw-bandwidth model —
        bigger objects are correctly *more* likely to persist when their
        recompute cost is high.
        """
        if execution_time < self._min_persist_compute_s:
            return False
        # Lazy import: only notebook-cached values carry the cost-model family,
        # and by then cash.notebook is already loaded. Keeps this module (and a
        # bare install / the decorator path) free of the notebook import.
        from cash.notebook import cost_model

        est_restore = cost_model.estimated_restore_time(type_name, size_bytes, backend_kind)
        return execution_time - est_restore > self._min_persist_savings_pct * execution_time

    def _default_promotion_policy(self, execution_time: float, size_bytes: int) -> bool:
        """Fallback policy used when no ``promotion_policy`` is supplied and the
        entry's metadata carries no cost-model family (so ``set`` can't predict
        with the real type).

        Serialization-aware like the smart policy, but assumes the slowest
        (``_GENERIC``) family since the caller gave only ``size_bytes``. Keeps
        the 1.0 s compute floor as a designed floor for the fallback path.
        """
        return self._cost_model_promote(
            "", size_bytes, execution_time, self._promotion_backend_kind()
        )

    def _warn_oversize_not_persisted(self, key: str, size_bytes: int) -> None:
        """Warn once/session that a worth-persisting value fit no disk tier.

        The object is larger than a safe fraction (half) of every persistent
        tier's cap, so persisting it would thrash. Cash offers it to the RAM
        tier rather than write-and-evict forever, and tells the user how to
        actually cache it. Deduped to once per session.

        "Offers", not "keeps": the RAM tier applies its own byte cap, which is
        machine-scaled and independent of ``max_cache_size``. That cap is
        normally SMALLER than the disk threshold that refused this value (4.0
        GiB against 18.7 GiB on one measured machine), so the usual outcome is
        eviction inside the same ``set()`` and no caching at all -- not
        RAM-only caching. Measured on a 4 MiB value that warns in both arms:
        RAM cap 100 MiB -> 2 calls, 1 execution; RAM cap 1 MiB -> 2 calls, 2
        executions."""
        if self._warned_oversize:
            return
        self._warned_oversize = True
        from cash.diagnostics import warn_diagnostic
        from cash.exceptions import CashCacheIneffectiveWarning
        from .adaptive_caps import human_bytes
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-VALUE-TOO-BIG",
            # "offered to the RAM tier", not "kept in RAM": the RAM tier applies
            # its own byte cap (InMemoryBackend._evict_to_byte_cap), which is
            # machine-scaled and independent of max_cache_size, and is normally
            # SMALLER than the disk threshold that refused this value -- so the
            # usual outcome is eviction inside the same set(), i.e. no caching
            # at all rather than RAM-only caching. Measured on a 4 MiB value:
            # RAM cap 100 MiB -> 2 calls, 1 execution; RAM cap 1 MiB -> 2 calls,
            # 2 executions. Keep this and docs/warnings.md#cache-value-too-big
            # saying the same thing.
            f"cached value {key!r} ({human_bytes(size_bytes)}) exceeds a safe "
            f"fraction of every persistent cache tier's cap, so only the RAM "
            f"tier was offered it -- it will not survive a kernel restart, and "
            f"if it is over the RAM tier's own cap too it is evicted at once "
            f"and nothing is cached.",
            "raise max_cache_size to a comfortable multiple of that size, or "
            "cache something smaller -- the aggregate, the sample, or the "
            "columns you actually use.",
            stacklevel=3,
        )

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

            # Promotion decision — same rule as the statement processor's Gate A
            # (predicted restore vs compute), applied here so the two gates can
            # never contradict each other. When the entry carries a cost-model
            # family (notebook-cached values), predict restore time with the
            # real type; otherwise fall through to the 2-arg promotion_policy
            # (injected test lambdas, the decorator path, legacy metadata).
            family = metadata.get('cost_model_family')
            if force_persist:
                past_compute_floor = True
            elif family is not None:
                past_compute_floor = self._cost_model_promote(
                    metadata.get('cost_model_type_name', ''),
                    metadata.get('cost_model_size_bytes', size),
                    exec_time,
                    self._promotion_backend_kind(),
                )
            else:
                past_compute_floor = self.promotion_policy(exec_time, size)

            # Size used for per-tier caps — prefer the cost-model estimate
            # (the notebook path sets no plain 'size' key).
            cap_size = size or metadata.get('cost_model_size_bytes', 0)

            size_refused = False  # a tier skipped this object because it's too big
            for i in range(1, len(self.backends)):
                backend = self.backends[i]
                if not past_compute_floor:
                    continue
                cap = backend._promotion_size_cap()
                # Guard against non-numeric caps (e.g. a MagicMock tier in
                # tests) — treat anything that isn't a real number as no cap.
                if isinstance(cap, bool) or not isinstance(cap, (int, float)):
                    cap = None
                if cap is not None and cap_size and cap_size > cap:
                    # This tier doesn't want objects this large. For the disk
                    # tier that means "bigger than half my cap" — storing it
                    # would thrash, so a clean skip beats the treadmill.
                    logger.debug(
                        "[TIERED] Skipping %s for key %r: size %d > cap %d",
                        type(backend).__name__, key, cap_size, cap,
                    )
                    size_refused = True
                    continue
                try:
                    backend.set(key, value, metadata, serializer)
                    _label = getattr(type(backend), 'source_label', None) or type(backend).__name__
                    stored_destinations.append(_label)
                except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
                    logger.warning("[TIERED] Failed to write to backend %s: %s", type(backend).__name__, e)

            # The value was worth persisting (cleared the compute floor) but
            # every persistent tier refused it as too big for its cap — it will
            # live in RAM only and vanish on the next kernel restart. A clean
            # no-op beats a treadmill, but the user should know why nothing
            # durable was written and how to fix it.
            if size_refused and not any(d != "RAM" for d in stored_destinations):
                self._warn_oversize_not_persisted(key, cap_size)

        # Update metadata with storage info so UI can see it immediately
        if metadata is not None:
             metadata['storage'] = stored_destinations

        # Propagate storage info back to the caller's original metadata dict
        if original_metadata is not None:
            original_metadata['storage'] = stored_destinations

        # Log visibility
        if stored_destinations:
            logger.debug("[STORAGE] Stored in: %s", ', '.join(stored_destinations))

