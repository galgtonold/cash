"""The statements whose next run collects the garbage first, kept across
kernels.

A statement whose only unseen writer ran inside a garbage collection is
not stored that once, and its next run collects first
(`StatementProcessor._collect_first`). Known in this session alone, a
notebook run top to bottom after each restart never reached the next run:
the statement was refused in every session. Kept here, the first run after
a restart collects first.

A collection before a statement only frees, sooner, what plain Python
frees at a moment no code chooses: a flag no longer needed costs one
collection, never a wrong answer.
"""

from __future__ import annotations

from ..versioned_json_store import VersionedJsonStore

__all__ = ["CollectFirstStore"]

#: Oldest statements go first past this many: the file is rewritten on
#: every change.
MAX_STATEMENTS = 1024


class CollectFirstStore(VersionedJsonStore[bool]):
    """``statement text digest:occurrence`` of each statement whose next run
    collects first (`processor._collect_place`)."""

    FILENAME = "collect_first.json"
    VERSION = 1
    FIELD = "places"
    LOG_TAG = "COLLECT_FIRST"

    def _load_value(self, value: object) -> bool | None:
        return True if value is True else None

    def __contains__(self, place: str) -> bool:
        self._ensure_loaded()
        return place in self._items

    def add(self, place: str) -> None:
        """Flag *place*, written out at once: a restart kills the kernel
        without an exit hook."""
        self._ensure_loaded()
        if place in self._items:
            return
        self._items[place] = True
        while len(self._items) > MAX_STATEMENTS:
            del self._items[next(iter(self._items))]
        self._write()

    def discard(self, place: str) -> None:
        """Take *place*'s flag off, once its run collected first."""
        self._ensure_loaded()
        if self._items.pop(place, None) is not None:
            self._write()
