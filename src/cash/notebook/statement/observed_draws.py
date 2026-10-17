"""The statements seen drawing from an RNG their text does not name, kept
across kernels.

``mean(solve(t) for t in suite)`` with a ``random.sample`` inside ``solve``
draws from ``random`` while its AST says nothing: the runtime learns it when
the statement runs (``observed_rng_statement_draws``) and stores the entry
under a key that holds the stream. A restart forgot what it learnt, so the
next run looked the statement up under the key without the stream, missed,
ran again and stored under the key it missed with: one recompute per
statement per restart, 50 s a time for the TSP benchmark of the
correctness set. Kept here, the first lookup after a restart builds the
key the entry was stored under.

Knowing of a draw that is no longer there only puts the stream in a key
that did not need it: a miss, never a wrong hit.
"""

from __future__ import annotations

from ..versioned_json_store import VersionedJsonStore

__all__ = ["ObservedDrawStore"]

#: Oldest statements go first past this many: the file is rewritten on
#: every new one.
_MAX_STATEMENTS = 20_000


class ObservedDrawStore(VersionedJsonStore[frozenset[str]]):
    """``exact source digest -> the RNG modules it was seen drawing from``."""

    FILENAME = "observed_draws.json"
    VERSION = 1
    FIELD = "draws"
    LOG_TAG = "OBSERVED_DRAWS"

    def _load_value(self, value: object) -> frozenset[str] | None:
        if isinstance(value, list) and value and all(isinstance(m, str) for m in value):
            return frozenset(value)
        return None

    def _dump_value(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    def draws(self) -> dict[str, frozenset[str]]:
        self._ensure_loaded()
        return dict(self._items)

    def record(self, digest: str, modules: set[str]) -> None:
        """Keep what *digest* draws from, written out at once: a restart
        kills the kernel without an exit hook."""
        self._ensure_loaded()
        if self._items.get(digest) == modules:
            return
        self._items.pop(digest, None)
        self._items[digest] = frozenset(modules)
        while len(self._items) > _MAX_STATEMENTS:
            del self._items[next(iter(self._items))]
        self._write()
