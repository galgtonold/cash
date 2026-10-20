"""Early cutoff: a statement that recomputed a value it already made once
gives that value its old lineage back.

A lineage names a value by how it was built. Rebuild a table with the same
code from an input that moved on (``temptastic = tsimp[...]`` after
``tsimp`` was rebuilt above it) and the new lineage differs although the
table holds the same rows -- so a 150 s loop reading it ran again in a
student's notebook. When a frame, series, array or networkx graph a statement
executed to bind has the same content as one an earlier statement bound, its
output takes that one's lineage, and what was cached from it is served. So
does an estimator, bound or fitted in place (`_refitted`).

Content is compared by `compute_hash`, which reads the whole value: values,
index, column names, dtypes and ``attrs``; for a graph its class, nodes and
edges in order and every attribute (`_GRAPHS`). A digest that is only an
identity (`is_identity_fallback_hash`) is never compared.

The alias from the lineage the formula gives to the one taken is kept in
`lineage_formula`, so the simulation, which recomputes lineages from code,
arrives at the same name, and in the cache directory
(`LineageAliasStore`): what was stored under the alias is found again by
the lineage a restarted kernel computes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any

from cash._user_attrs import attribute_or

from .._plain_data import is_plain
from ..analysis.namespace_effects import is_estimator
from ..value_hash import digest_via_pickle, is_identity_fallback_hash
from ..value_types import IMMUTABLE_LEAF_TYPES
from .lineage_formula import record_lineage_alias, resolve_lineage_alias
from .versioned_json_store import VersionedJsonStore

__all__ = ["LineageAliasStore", "ValueCutoff"]

#: Aliases kept on disk at most; the oldest go first.
_MAX_ALIASES = 50_000

#: The types compared: a frame, series or array, whose content hash covers
#: everything a reader can see of it, and a networkx graph, whose content
#: hash is the key pickle of all it holds (`_GRAPHS`). Other values keep
#: their own lineage.
_COMPARED = {
    ("pandas", "DataFrame"),
    ("pandas", "Series"),
    ("numpy", "ndarray"),
    ("networkx", "Graph"),
    ("networkx", "DiGraph"),
    ("networkx", "MultiGraph"),
    ("networkx", "MultiDiGraph"),
}

#: A networkx graph's content hash (`compute_hash`) is the digest of its key
#: pickle: its class by name, and its ``__dict__`` -- the graph, node and
#: adjacency dicts in their order (so node order and, per node, edge order),
#: every attribute dict and every value in it by content, and, for a view
#: (``G.subgraph(nodes)``), the graph it views and its filters. Only the
#: views networkx fills on first use (``G.nodes``, ``G.adj``) are left out
#: (`kept_state.derived_views`): they hold nothing the graph does not. Equal
#: graphs whose objects are shared otherwise (one attribute dict on two
#: nodes) pickle apart: a miss, never a wrong lineage. The digest is taken
#: once per statement: the session hash of the output reuses it
#: (`StatementLineageBuilder._hash_once`). A graph keeps no version counter,
#: so it is not carried to the next statement.
_GRAPHS = frozenset(name for module, name in _COMPARED if module == "networkx")


def _compared(value: Any) -> bool:
    t = type(value)
    if ((t.__module__ or "").split(".", 1)[0], t.__name__) not in _COMPARED:
        return False
    if t.__name__ in _GRAPHS:
        return True
    # An array of objects (``axes`` from ``plt.subplots``) is hashed by
    # pickling what it holds -- a whole figure.
    return attribute_or(attribute_or(value, "dtype", None), "kind", "") != "O"


def _refitted(value: Any) -> bool:
    """An estimator (`is_estimator`) whose digest is its whole pickle
    (`digest_via_pickle`): the bytes a restore gives back, so an equal digest
    is an equal model. ``model.fit(X, y)`` run again on the model it already
    fitted gave it a new lineage each run, so what was built on it missed and
    the next identical fit ran again: a held-out session re-ran a 95 s
    CatBoost fit five times. The statement compares it even though it
    changed the model in place (`StatementLineageBuilder`).
    """
    return is_estimator(value) and digest_via_pickle(value)


def _is_plain_value(value: Any) -> bool:
    """A scalar, or exact lists and tuples of them (`_plain_data.is_plain`):
    its content digest is all there is to it."""
    if type(value) in IMMUTABLE_LEAF_TYPES:
        return True
    return type(value) in (list, tuple) and is_plain(value)


def _read_only(value: Any) -> str:
    """Which of *value*'s arrays cannot be written: an array equal to an
    earlier one but read-only (``df["a"].to_numpy()`` in pandas 3) behaves
    otherwise (``arr[arr < 2] = 0`` raises), so it is not the same value.
    One flag per array: ``"1"`` read-only, ``"0"`` writeable, ``"?"`` not
    told."""
    numpy = sys.modules.get("numpy")
    if numpy is None or type(value).__name__ in _GRAPHS:
        return ""
    if isinstance(value, numpy.ndarray):
        arrays = [value]
    else:
        try:
            arrays = list(value._mgr.arrays)  # a series' or frame's blocks
        except Exception:  # noqa: BLE001 - not listed: told by what is seen
            return "?"
    return "".join(("0" if a.flags.writeable else "1") if isinstance(a, numpy.ndarray) else "?" for a in arrays)


class LineageAliasStore(VersionedJsonStore[str]):
    """``lineage -> the earlier lineage its value equals``, in the cache dir."""

    FILENAME = "lineage_aliases.json"
    #: 2: aliases of a statement with several outputs are no longer made.
    VERSION = 2
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
    """Content digest -> the lineage of the first value seen with it.

    Seen executed (`lineage_for`), or restored from an entry that kept the
    digest its value had when it ran (`remember`): after a restart most
    values come back from the cache, and a value only restored was never
    seen, so recomputing it from inputs that moved on and back (a frame
    filtered to the rows it held before) missed what was built on it.
    """

    def __init__(self) -> None:
        self._by_digest: dict[str, str] = {}
        #: ``{(name, digest), ...}`` of every output of a statement that bound
        #: several -> the lineage they shared (`lineage_for_outputs`).
        self._by_outputs: dict[frozenset[tuple[str, str]], str] = {}
        #: Lineage -> the digest of the value it names, for the entry stored
        #: with that value (`digest_of`).
        self._digest_of: dict[str, str] = {}

    def clear(self) -> None:
        """Forget every value seen, as a fresh kernel would."""
        self._by_digest.clear()
        self._digest_of.clear()
        self._by_outputs.clear()

    def __len__(self) -> int:
        return len(self._by_digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValueCutoff):
            return NotImplemented
        return (
            self._by_digest == other._by_digest
            and self._digest_of == other._digest_of
            and self._by_outputs == other._by_outputs
        )

    __hash__ = None  # type: ignore[assignment]  # mutable

    def digest_of(self, lineage: str) -> str | None:
        """The digest of the value *lineage* names, when one was seen."""
        return self._digest_of.get(lineage)

    def remember(self, digest: str, lineage: str) -> None:
        """Note that a value with *digest* (`digest_of`, kept on a cache
        entry) was restored with *lineage*. An equal value seen first keeps
        its lineage."""
        self._by_digest.setdefault(digest, resolve_lineage_alias(lineage))
        self._digest_of.setdefault(lineage, digest)

    def lineage_for_outputs(
        self, values: Mapping[str, Any], lineages: Mapping[str, str], compute_hash: Callable[[Any], str]
    ) -> dict[str, str]:
        """The lineages the outputs of one statement binding several values,
        just bound with *lineages*, should carry.

        They share one formula lineage, so a cutoff of one of them would
        name the others' values too (``train, test = ...`` with only
        ``train`` unchanged). All of them together take an earlier lineage
        only when an earlier statement bound the same names to values with
        the same content and gave them one lineage: ``U, V, hist =
        als_fit(C, tol=1e-3)`` returning what ``als_fit(C)`` did keeps what
        was built on ``U`` and ``V``. Each value must be a frame, series,
        array or networkx graph (`_compared`), or plain data
        (`_plain_data.is_plain`), whose digest covers all of it; otherwise
        *lineages* stand.
        """
        shared = set(lineages.values())
        if len(shared) != 1 or set(values) != set(lineages):
            return dict(lineages)
        (lineage,) = shared
        parts: set[tuple[str, str]] = set()
        for name, value in values.items():
            digest = self._digest(value, compute_hash, plain=True)
            if digest is None:
                return dict(lineages)
            parts.add((name, digest))
        outputs = frozenset(parts)
        earlier = self._by_outputs.get(outputs)
        if earlier is None:
            self._by_outputs[outputs] = resolve_lineage_alias(lineage)
            return dict(lineages)
        earlier = resolve_lineage_alias(earlier)
        if earlier != lineage:
            record_lineage_alias(lineage, earlier)
        return dict.fromkeys(lineages, earlier)

    @staticmethod
    def _digest(value: Any, compute_hash: Callable[[Any], str], plain: bool = False) -> str | None:
        """*value*'s typed content digest, or None when it is not compared
        (`_compared`, `_refitted`; with *plain*, plain data too) or has no
        content hash."""
        if not (_compared(value) or _refitted(value) or (plain and _is_plain_value(value))):
            return None
        try:
            digest = compute_hash(value)
        except Exception:  # noqa: BLE001 - an unhashable value keeps its lineage
            return None
        if is_identity_fallback_hash(value, digest):
            return None
        return f"{type(value).__name__}:{_read_only(value)}:{digest}"

    def lineage_for(self, value: Any, lineage: str, compute_hash: Callable[[Any], str]) -> str:
        """The lineage *value*, just bound with *lineage*, should carry: an
        earlier value's when its content is the same, else *lineage*."""
        digest = self._digest(value, compute_hash)
        if digest is None:
            return lineage
        earlier = self._by_digest.get(digest)
        if earlier is None:
            self._by_digest[digest] = resolve_lineage_alias(lineage)
            self._digest_of[lineage] = digest
            return lineage
        earlier = resolve_lineage_alias(earlier)
        if earlier != lineage:
            record_lineage_alias(lineage, earlier)
        self._digest_of[earlier] = digest
        return earlier
