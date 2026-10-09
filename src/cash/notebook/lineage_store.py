"""The single seam for reading and writing variable lineage.

Every statement cache key is built in one place (``compute_cache_key``);
this module applies the same rule to *lineage state*: one place writes it.

Invariants
----------
* Every lineage write goes through :meth:`LineageStore.record`,
  :meth:`LineageStore.reset_to`, :meth:`LineageStore.discard` or
  :meth:`LineageStore.clear`. Readers get a read-only view
  (:attr:`LineageStore.view`, which ``TrackingState.variable_lineage``
  returns), so a direct dict write fails loudly instead of skipping the tag.
* When ``record`` is given a ``value`` that can be tagged, the entry and the
  value's tag (`tag_value`, held beside the value) are written together so
  they cannot drift.
* The priority ladder lives in :func:`resolve_lineage`: virtual → store →
  the value's own tag (`own_tag`) → ``compute_hash_fn`` → ``sha256(str(value))``.
"""

from __future__ import annotations

import hashlib
import logging
import weakref
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from cash.lineage_tag import clear_tags, own_tag, set_tags

logger = logging.getLogger(__name__)

__all__ = ["InputLineages", "LineageStore", "LineageWatch", "resolve_lineage", "tag_value"]


def tag_value(value: Any, hash_: str) -> None:
    """Tag *value* with the lineage *hash_* a statement gave it, beside it.

    The tag lives in `cash.lineage_tag`'s side table, never on the object:
    written into its ``__dict__``, it showed up in the user's own data
    (``vars(args)``, ``json.dumps(vars(cfg))``, a ``SimpleNamespace``
    comparison, the object's pickle). Never a class, module or function
    either: a tag on a class is inherited by every instance, which then all
    key alike. Values that take no tag (builtins, slotted types) keep only
    the store's entry, which stays authoritative.
    """
    # Drop any tag the decorator left (its producer too): this newer tag wins.
    clear_tags(value)
    # This layer re-tags the value whenever it changes, which is what lets
    # the decorator trust the tag for its content ("statement").
    if not set_tags(value, _cash_lineage_hash=hash_, _cash_lineage_src="statement"):
        logger.debug("LineageStore: cannot tag %s", type(value).__name__)


def resolve_lineage(
    var: str,
    lineage: Mapping[str, str],
    *,
    value: Any,
    virtual: Mapping[str, str] | None = None,
    compute_hash_fn: Callable[[Any], str] | None = None,
) -> str | None:
    """Resolve a lineage hash for *var* using the priority ladder.

    Order: ``virtual`` mapping → *lineage* → the value's own tag (`own_tag`) →
    ``compute_hash_fn(value)`` → ``sha256(str(value))``.
    """
    if virtual is not None and var in virtual:
        return virtual[var]
    if var in lineage:
        return lineage[var]
    if value is None:
        return None
    try:
        attr = own_tag(value)
        if attr is not None:
            return attr
        if compute_hash_fn is not None:
            return compute_hash_fn(value)
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, RecursionError):
        logger.debug("LineageStore: failed to compute lineage for %r", var)
        return None


#: A name a watch saw with no lineage, or no input map, before its change.
_ABSENT = object()


class LineageWatch:
    """What changed since the watch began (`LineageStore.watch`).

    For each name written since -- its lineage or its input map -- what it
    held just before the first such write (``_ABSENT`` when it had none).
    The upstream check asks which names its repair recorded again: a walk
    over every name the notebook binds, on every cell run, before. Here it
    is the few names written. The old input maps are held, so ``is`` against
    them never meets a reused id.

    The store holds a watch weakly: one dropped without being read stops
    collecting.
    """

    __slots__ = ("inputs", "lineages", "__weakref__")

    def __init__(self) -> None:
        self.lineages: dict[str, object] = {}
        self.inputs: dict[str, object] = {}

    def rerecorded(self, lineage: Mapping[str, str], inputs: Mapping[str, dict[str, str]]) -> set[str]:
        """The names holding a lineage now whose lineage is new or another,
        or whose input map is not the one held when the watch began."""
        changed = set()
        for name in self.lineages.keys() | self.inputs.keys():
            now = lineage.get(name)
            if now is None:
                continue
            if self.lineages.get(name, now) != now:
                changed.add(name)
            elif name in self.inputs and self.inputs[name] is not inputs.get(name, _ABSENT):
                changed.add(name)
        return changed


def _note(watches: weakref.WeakSet, field: str, held: Mapping, names) -> None:
    """Note in each watch what *names* hold in *held* before their change."""
    for watch in watches:
        seen = getattr(watch, field)
        for name in names:
            if name not in seen:
                seen[name] = held.get(name, _ABSENT)


class InputLineages(dict):
    """``executed_input_lineages``: a plain dict whose writes the store's
    watches see (`LineageWatch`). Reads are a dict's."""

    __slots__ = ("_watches",)

    def __init__(self, watches: weakref.WeakSet, *args: Any, **kwargs: Any) -> None:
        self._watches = watches
        super().__init__(*args, **kwargs)

    def __reduce__(self):
        # A copy is a plain dict: no watch is about it.
        return (dict, (dict(self),))

    def _changing(self, names) -> None:
        if self._watches:
            _note(self._watches, "inputs", self, names)

    def __setitem__(self, key, value) -> None:
        self._changing((key,))
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        self._changing((key,))
        super().__delitem__(key)

    def pop(self, key, *default):
        self._changing((key,))
        return super().pop(key, *default)

    def popitem(self):
        key, value = super().popitem()
        if self._watches:
            for watch in self._watches:
                watch.inputs.setdefault(key, value)
        return key, value

    def setdefault(self, key, default=None):
        if key not in self:
            self._changing((key,))
        return super().setdefault(key, default)

    def update(self, *args, **kwargs) -> None:
        other = dict(*args, **kwargs)
        self._changing(other)
        super().update(other)

    def __ior__(self, other):
        self.update(other)
        return self

    def clear(self) -> None:
        self._changing(list(self))
        super().clear()


class LineageStore(Mapping[str, str]):
    """Owns every variable's lineage hash and the paired value tag. Reads through the ``Mapping`` interface or :attr:`view`; writes
    only through the methods below. See the module docstring for invariants."""

    def __init__(self) -> None:
        self._lineage: dict[str, str] = {}
        #: A read-only live view of the store, for readers that want a mapping.
        self.view: Mapping[str, str] = MappingProxyType(self._lineage)
        #: The watches open now (`watch`); shared with `input_lineages`.
        self._watches: weakref.WeakSet[LineageWatch] = weakref.WeakSet()

    def watch(self) -> LineageWatch:
        """Start noting what changes, here and in the maps of
        `input_lineages`, until the returned watch is dropped."""
        watch = LineageWatch()
        self._watches.add(watch)
        return watch

    def input_lineages(self, initial: Mapping[str, dict[str, str]] = ()) -> InputLineages:
        """A map of input lineages whose writes this store's watches see."""
        return InputLineages(self._watches, initial)

    def _changing(self, names) -> None:
        if self._watches:
            _note(self._watches, "lineages", self._lineage, names)

    # --- read surface ---------------------------------------------------

    def __getitem__(self, var: str) -> str:
        return self._lineage[var]

    def __contains__(self, var: object) -> bool:
        return var in self._lineage

    def __iter__(self) -> Iterator[str]:
        return iter(self._lineage)

    def __len__(self) -> int:
        return len(self._lineage)

    # --- write surface --------------------------------------------------

    def record(self, var: str, hash_: str, *, value: Any = None) -> None:
        """Record the persistent lineage for *var*.

        When *value* is provided, also tag it (`tag_value`) so the entry and
        the tag cannot drift.
        """
        self._changing((var,))
        self._lineage[var] = hash_
        if value is not None:
            tag_value(value, hash_)

    def reset_to(self, var: str, hash_: str) -> None:
        """Resynchronise *var*'s lineage to *hash_* without touching the value.

        Used by the simulator when downstream advancement leaves the recorded
        lineage 'ahead' of where simulation says it should be. Distinct from
        :meth:`record` because no fresh computation happened — only state-machine
        correction. The value's tag reflects the value's
        actual computation and must NOT be rewritten here.
        """
        self._changing((var,))
        self._lineage[var] = hash_

    def discard(self, var: str) -> None:
        """Forget *var*'s lineage, if it has one."""
        self._changing((var,))
        self._lineage.pop(var, None)

    def clear(self) -> None:
        """Forget every lineage."""
        self._changing(list(self._lineage))
        self._lineage.clear()

    # --- priority ladder ------------------------------------------------

    def resolve(
        self,
        var: str,
        *,
        value: Any,
        virtual: Mapping[str, str] | None = None,
        compute_hash_fn: Callable[[Any], str] | None = None,
    ) -> str | None:
        """:func:`resolve_lineage` against this store."""
        return resolve_lineage(var, self._lineage, value=value, virtual=virtual, compute_hash_fn=compute_hash_fn)
