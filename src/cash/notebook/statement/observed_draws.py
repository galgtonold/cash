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

__all__ = ["AdoptedSeedStore", "ObservedDrawStore"]

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


class AdoptedSeedStore(VersionedJsonStore[tuple[str, dict[str, str]]]):
    """``exact source digest -> (its key, {RNG variable: lineage})``: what a
    statement seeded through code cash does not read as seeding
    (``mylib.setup(4)``, a seed in ``__init__``), as ``adopted_rng_after``
    keeps it. A restart forgot it, so a cell run alone after one keyed and
    rebuilt the draws below such a seed as if the stream were unseeded.

    The lineage is a function of where the seed put the stream, the same in
    every kernel; the key is checked before it is used, so a statement that
    changed since takes nothing from here.
    """

    FILENAME = "adopted_seeds.json"
    VERSION = 1
    FIELD = "seeds"
    LOG_TAG = "ADOPTED_SEEDS"

    def _load_value(self, value: object) -> tuple[str, dict[str, str]] | None:
        if (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in value[1].items())
        ):
            return value[0], dict(value[1])
        return None

    def _dump_value(self, value: tuple[str, dict[str, str]]) -> list:
        return [value[0], dict(sorted(value[1].items()))]

    def seeds(self) -> dict[str, tuple[str, dict[str, str]]]:
        self._ensure_loaded()
        return dict(self._items)

    def record(self, digest: str, entry: tuple[str, dict[str, str]] | None) -> None:
        """Keep *entry* for *digest* (None: forget it), written out at once."""
        self._ensure_loaded()
        if self._items.get(digest) == entry:
            return
        self._items.pop(digest, None)
        if entry is not None:
            self._items[digest] = entry
            while len(self._items) > _MAX_STATEMENTS:
                del self._items[next(iter(self._items))]
        self._write()
