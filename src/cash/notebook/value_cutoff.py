"""Early cutoff: a statement that recomputed a value it already made once
gives that value its old lineage back.

A lineage names a value by how it was built. Rebuild a table with the same
code from an input that moved on (``temptastic = tsimp[...]`` after
``tsimp`` was rebuilt above it) and the new lineage differs although the
table holds the same rows -- so a 150 s loop reading it ran again in a
student's notebook. When a frame, series or array a statement executed to
bind has the same content as one an earlier statement bound, its output takes
that one's lineage, and what was cached from it is served.

Content is compared by `compute_hash`, which reads the whole value: values,
index, column names, dtypes and ``attrs``. A digest that is only an identity
(`is_identity_fallback_hash`) is never compared.

The alias from the lineage the formula gives to the one taken is kept in
`lineage_formula`, so the simulation, which recomputes lineages from code,
arrives at the same name, and in the cache directory
(`LineageAliasStore`): what was stored under the alias is found again by
the lineage a restarted kernel computes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..value_hash import is_identity_fallback_hash
from .lineage_formula import record_lineage_alias, resolve_lineage_alias
from .versioned_json_store import VersionedJsonStore

__all__ = ["LineageAliasStore", "ValueCutoff"]

#: Aliases kept on disk at most; the oldest go first.
_MAX_ALIASES = 50_000

#: The types compared: a frame, series or array, whose content hash covers
#: everything a reader can see of it. Other values keep their own lineage.
_COMPARED = {("pandas", "DataFrame"), ("pandas", "Series"), ("numpy", "ndarray")}


def _compared(value: Any) -> bool:
    t = type(value)
    if ((t.__module__ or "").split(".", 1)[0], t.__name__) not in _COMPARED:
        return False
    # An array of objects (``axes`` from ``plt.subplots``) is hashed by
    # pickling what it holds -- a whole figure.
    return getattr(getattr(value, "dtype", None), "kind", "") != "O"


class LineageAliasStore(VersionedJsonStore[str]):
    """``lineage -> the earlier lineage its value equals``, in the cache dir."""

    FILENAME = "lineage_aliases.json"
    VERSION = 1
    FIELD = "aliases"
    LOG_TAG = "LINEAGE_ALIASES"

    def _load_value(self, value: object) -> str | None:
        return value if isinstance(value, str) else None

    def aliases(self) -> dict[str, str]:
        self._ensure_loaded()
        return dict(self._items)

    def record(self, lineage: str, earlier: str) -> None:
        """Keep one alias, written out at once: a cutoff is rare, and a
        restart kills the kernel without an exit hook."""
        self._ensure_loaded()
        self._items[lineage] = earlier
        while len(self._items) > _MAX_ALIASES:
            del self._items[next(iter(self._items))]
        self._write()


class ValueCutoff:
    """Content digest -> the lineage of the first value seen with it."""

    def __init__(self) -> None:
        self._by_digest: dict[str, str] = {}

    def lineage_for(self, value: Any, lineage: str, compute_hash: Callable[[Any], str]) -> str:
        """The lineage *value*, just bound with *lineage*, should carry: an
        earlier value's when its content is the same, else *lineage*."""
        if not _compared(value):
            return lineage
        try:
            digest = compute_hash(value)
        except Exception:  # noqa: BLE001 - an unhashable value keeps its lineage
            return lineage
        if is_identity_fallback_hash(value, digest):
            return lineage
        digest = f"{type(value).__name__}:{digest}"
        earlier = self._by_digest.get(digest)
        if earlier is None:
            self._by_digest[digest] = resolve_lineage_alias(lineage)
            return lineage
        earlier = resolve_lineage_alias(earlier)
        if earlier != lineage:
            record_lineage_alias(lineage, earlier)
        return earlier
