"""One decorated function: what it was decorated with, and what this process
has learned about its calls."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["CHUNK_MAX_BYTES", "CHUNK_MAX_ITEMS", "WARNINGS_MAX", "CachedFunction", "PurityMode", "new_stats"]

#: A returned iterator's chunk closes at whichever of these it reaches first.
CHUNK_MAX_ITEMS = 1_000_000
CHUNK_MAX_BYTES = 1_000_000_000

#: Recent warnings `cache_info()` reports per function.
WARNINGS_MAX = 20

#: ``strict`` raises on a purity issue, ``silent`` (``assume_safe=True``)
#: suppresses the warnings, ``warn`` is the default.
PurityMode = Literal["warn", "strict", "silent"]

#: `CachedFunction.signature` before it is first read.
_UNREAD: Any = object()


def new_stats() -> dict[str, Any]:
    """The counters `cache_info()` and the run summary read."""
    return {
        "hits": 0,
        "misses": 0,
        "total_time_saved": 0.0,
        "lookup_seconds": 0.0,
        # What the misses cost besides their bodies: key, checks, store.
        "miss_overhead_seconds": 0.0,
        # What the misses were, and which results did not reach disk.
        "miss_reasons": Counter(),
        "not_persisted": Counter(),
        "not_stored": Counter(),
        # For "code or state changed" misses: WHAT changed.
        "changed": Counter(),
        # Calls that went straight through because caching is off.
        "bypassed": 0,
    }


@dataclass(eq=False)
class CachedFunction:
    """The options of one ``@cache`` decoration and its per-process state.

    A name decorated again (a notebook cell re-run) gets a new record; what
    was learned about the name's calls rather than its code carries over
    (`carry_over`).
    """

    func: Callable
    name: str
    dynamic_depends_on: Any = None
    ttl: int | None = None
    cache_if: Callable[[Any], bool] | None = None
    chunk_max_items: int = CHUNK_MAX_ITEMS
    chunk_max_bytes: int = CHUNK_MAX_BYTES
    purity: PurityMode = "warn"
    frozen: bool = False
    allow_random: bool = False
    #: ``file_depends_on=`` as ``(as written, absolute)`` pairs.
    declared_files: tuple[tuple[str, str], ...] = ()

    #: The wrapper `Cash.cache` returned.
    wrapper: Callable | None = field(default=None, repr=False)
    stats: dict[str, Any] = field(default_factory=new_stats, repr=False)
    #: Recent warnings, newest last, at most `WARNINGS_MAX`.
    warnings: list[dict[str, Any]] = field(default_factory=list, repr=False)
    #: The key of the last call, for "what changed since the last call".
    last_key: str | None = None
    #: The costliest argument hash seen: ``(label, type, seconds, producer, old_pandas)``.
    arg_cost: tuple | None = None
    #: RNG modules its calls were seen drawing from; None until known.
    rng_modules: set[str] | None = None
    #: Seeding expressions fed by a parameter or a global (`seed_parameters`).
    seed_params: dict[str, tuple] = field(default_factory=dict, repr=False)
    #: The argument-mutation check cost more than its budget and is off.
    mutation_check_retired: bool = False
    _signature: Any = field(default=_UNREAD, repr=False)

    @property
    def signature(self) -> inspect.Signature | None:
        """``inspect.signature(func)``, read once; None when it has none."""
        if self._signature is _UNREAD:
            try:
                self._signature = inspect.signature(self.func)
            except (ValueError, TypeError):
                self._signature = None
        return self._signature

    def carry_over(self, previous: CachedFunction) -> None:
        """Keep what *previous*, the name's earlier definition, learned about
        its calls: the last key (so the next miss can say the code changed),
        the warning log, the argument cost, the RNG verdict."""
        self.last_key = previous.last_key
        self.warnings = previous.warnings
        self.arg_cost = previous.arg_cost
        self.rng_modules = previous.rng_modules
        self.mutation_check_retired = previous.mutation_check_retired
