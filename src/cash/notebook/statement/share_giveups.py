"""What the share check and the holder digests gave up on, remembered for
the same objects (`StatementProcessor._refuse_unrestorable_outputs`,
`StatementProcessor._holder_digests`).

A frame built from a list of 40,000 sessions shares every session's list
with that list (``df = pd.DataFrame(logs)``): the share check of ``df``
gives up on its budget, and asked again for each statement that changes
``df`` in place (``df['n'] = ...``), it reached the same verdict a second
later each time. Remembered, the verdict is taken again while the objects
are the same ones -- ``df`` and each variable found holding its objects
still bound to them, their lineage where the last verdict left it, the
arrays of ``df``'s object columns still its own, the edges between them
still recorded. Anything else asks again.

A verdict taken again only refuses: the statement is not stored, its links
are looked for as after a give-up. At worst a statement that could have
been stored this time runs once more uncached; nothing is served.

Only ids are kept, never the objects: the share check counts references.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from cash._user_attrs import attribute_or
from cash.notebook.statement.derivation_edges import shared_object_partners


def column_owners(value: Any) -> frozenset[tuple[Any, int]]:
    """``(column, id of the array owning its memory)`` for each of *value*'s
    object columns, when it is a pandas DataFrame; empty for anything else.

    By column, not by array alone: pandas 2 keeps the frame's object columns
    in one two-dimensional block, and ``df['actions'] = 0`` takes the column
    out of it while the column beside it stays a view of the same array."""
    if (attribute_or(type(value), "__module__", None) or "").partition(".")[0] != "pandas":
        return frozenset()
    mgr = attribute_or(value, "_mgr", None)
    if mgr is None:
        return frozenset()
    try:
        import numpy as np

        columns = value.columns
        owners = set()
        for block in mgr.blocks:
            values = block.values
            if type(values) is not np.ndarray or values.dtype != object:
                continue
            while type(values.base) is np.ndarray:
                values = values.base
            owners.update((columns[i], id(values)) for i in block.mgr_locs.as_array)
        return frozenset(owners)
    except Exception:  # noqa: BLE001 - not a frame's block manager after all
        return frozenset()


@dataclass
class _GaveUp:
    """One output's last give-up: ids of its object and object columns, and
    of each holder the search up found; the lineages once settled."""

    value: int
    columns: frozenset[tuple[Any, int]]
    holders: dict[str, int]
    lineage: dict[str, Any] | None = None


class ShareGiveUps:
    """The share check's last give-up for each output name.

    Noted when the check gave up on its budget and the search from the
    outputs up found variables holding the output's objects (`note`);
    settled with the lineages the statement left (`settle`). As the next
    statement starts, an entry stands only while its output and holders are
    bound to the same objects with those lineages and the edges between
    them are still recorded (`live`); its output then gives up again only
    while the statement neither rebinds it nor its holders, changes no
    holder, and leaves the output's object columns in place (`hit`).
    """

    def __init__(self) -> None:
        self._entries: dict[str, _GaveUp] = {}
        #: Noted or hit by the running statement: settled as it ends.
        self._noted: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def live(
        self, user_ns: Mapping[str, Any], lineage: Mapping[str, Any], edges: dict[str, set[str]]
    ) -> frozenset[str]:
        """As a statement starts: the outputs whose give-up still stands.
        The others are forgotten."""
        self._noted.clear()
        live = set()
        for name, entry in list(self._entries.items()):
            if (
                entry.lineage is not None
                and self._intact(name, entry, user_ns, edges)
                and self._same_lineage(entry, lineage)
            ):
                live.add(name)
            else:
                del self._entries[name]
        return frozenset(live)

    def hit(
        self,
        live: Collection[str],
        values: Mapping[str, Any],
        user_ns: Mapping[str, Any],
        bound: Collection[str],
        changed: Collection[str],
    ) -> bool:
        """Whether every output in *values* gives up again: each one *live*,
        neither it nor a holder of its bound by the statement (*bound*), no
        holder *changed* by it, and its object columns still its own."""
        if not values:
            return False
        for name, value in values.items():
            entry = self._entries.get(name)
            if name not in live or entry is None or name in bound or id(value) != entry.value:
                return False
            if any(holder in bound or holder in changed for holder in entry.holders):
                return False
            if any(id(user_ns.get(holder)) != held for holder, held in entry.holders.items()):
                return False
            if not entry.columns <= column_owners(value):
                return False
        self._noted.update(values)
        return True

    def note(self, name: str, value: Any, holders: Mapping[str, Any]) -> None:
        """*name*, bound to *value*, gave up with *holders* found holding its
        objects: its entry, replacing the last; settled as the statement
        ends."""
        self._entries[name] = _GaveUp(id(value), column_owners(value), {h: id(v) for h, v in holders.items()})
        self._noted.add(name)

    def keep(self, name: str, value: Any) -> None:
        """*name* gave up again, bound to *value*, and the search up was cut
        short before it found any holder: its entry stands with the holders
        the last give-up found, when it is for *value*; forgotten else."""
        entry = self._entries.get(name)
        if entry is None or entry.value != id(value):
            self.forget([name])
        else:
            entry.columns = column_owners(value)
            self._noted.add(name)

    def forget(self, names: Iterable[str]) -> None:
        """The share check did not give up on *names* this time."""
        for name in names:
            self._entries.pop(name, None)
            self._noted.discard(name)

    def settle(self, user_ns: Mapping[str, Any], lineage: Mapping[str, Any], edges: dict[str, set[str]]) -> None:
        """As the statement ends: each entry it noted or hit takes the
        lineages it left, or is forgotten when its objects or edges are not
        the ones noted any more."""
        for name in self._noted:
            entry = self._entries.get(name)
            if entry is None:
                continue
            if self._intact(name, entry, user_ns, edges):
                entry.lineage = {other: lineage.get(other) for other in (name, *entry.holders)}
            else:
                del self._entries[name]
        self._noted.clear()

    @staticmethod
    def _intact(name: str, entry: _GaveUp, user_ns: Mapping[str, Any], edges: dict[str, set[str]]) -> bool:
        if name not in user_ns or id(user_ns[name]) != entry.value:
            return False
        if any(holder not in user_ns or id(user_ns[holder]) != held for holder, held in entry.holders.items()):
            return False
        return set(entry.holders) <= shared_object_partners(edges, name)

    @staticmethod
    def _same_lineage(entry: _GaveUp, lineage: Mapping[str, Any]) -> bool:
        assert entry.lineage is not None
        return all(lineage.get(name) == kept for name, kept in entry.lineage.items())


class CostlyDigests:
    """The holders whose in-place-edit digest did not finish within its
    budget (`StatementProcessor._holder_digests`), by name: not asked again
    while still bound to the same object with the lineage the statement
    that gave up left. Such a holder moves on, as one whose digest ran out
    of time does; a wrong value cannot come of it, only a holder moved on
    that the digest would have shown unchanged."""

    def __init__(self) -> None:
        #: name -> (id of its object, its lineage), the lineage None until settled.
        self._entries: dict[str, tuple[int, Any]] = {}
        self._noted: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def begin(self) -> None:
        """A statement's holder digests start."""
        self._noted.clear()

    def known(self, name: str, value: Any, lineage: Any) -> bool:
        """Whether *name*'s digest is known too costly for *value* at
        *lineage*; an entry that is not for them is forgotten."""
        entry = self._entries.get(name)
        if entry is None:
            return False
        if entry[1] is not None and entry == (id(value), lineage):
            self._noted.add(name)
            return True
        del self._entries[name]
        return False

    def note(self, name: str, value: Any) -> None:
        """*name*'s digest of *value* ran out of time."""
        self._entries[name] = (id(value), None)
        self._noted.add(name)

    def settle(self, user_ns: Mapping[str, Any], lineage: Mapping[str, Any], changed: Collection[str]) -> None:
        """As the statement ends: each holder it noted or skipped keeps its
        entry with the lineage it left, unless the statement *changed* it
        itself or it is bound to another object now."""
        for name in self._noted:
            entry = self._entries.get(name)
            if entry is None:
                continue
            if name in changed or name not in user_ns or id(user_ns[name]) != entry[0]:
                del self._entries[name]
            else:
                self._entries[name] = (entry[0], lineage.get(name))
        self._noted.clear()
