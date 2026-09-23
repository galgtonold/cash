"""The single seam for reading and writing variable lineage.

See ``docs/architecture_decisions.md`` ADR-007 for the *cache-key* invariant
this module extends to *lineage state*.

Invariants
----------
* Every lineage write goes through :meth:`LineageStore.record`,
  :meth:`LineageStore.reset_to`, :meth:`LineageStore.discard` or
  :meth:`LineageStore.clear`. Readers get a read-only view
  (:attr:`LineageStore.view`, which ``TrackingState.variable_lineage``
  returns), so a direct dict write fails loudly instead of skipping the tag.
* When ``record`` is given a ``value`` whose type accepts attributes, the
  entry and the value's ``_cash_lineage_hash`` attribute are written together
  so they cannot drift.
* The priority ladder lives in :func:`resolve_lineage`: virtual → store →
  ``value._cash_lineage_hash`` → ``compute_hash_fn`` → ``sha256(str(value))``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from cash.lineage_tag import own_tag, taggable

logger = logging.getLogger(__name__)

__all__ = ["LineageStore", "resolve_lineage"]


def resolve_lineage(
    var: str,
    lineage: Mapping[str, str],
    *,
    value: Any,
    virtual: Mapping[str, str] | None = None,
    compute_hash_fn: Callable[[Any], str] | None = None,
) -> str | None:
    """Resolve a lineage hash for *var* using the priority ladder.

    Order: ``virtual`` mapping → *lineage* → ``value._cash_lineage_hash`` →
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


class LineageStore(Mapping[str, str]):
    """Owns every variable's lineage hash and the paired ``_cash_lineage_hash``
    attribute. Reads through the ``Mapping`` interface or :attr:`view`; writes
    only through the methods below. See the module docstring for invariants."""

    def __init__(self) -> None:
        self._lineage: dict[str, str] = {}
        #: A read-only live view of the store, for readers that want a mapping.
        self.view: Mapping[str, str] = MappingProxyType(self._lineage)

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

        When *value* is provided and accepts attributes, also set
        ``value._cash_lineage_hash`` so the entry and the attribute cannot drift.
        """
        self._lineage[var] = hash_
        # Never a class, module or function: a tag on a class is inherited by
        # every instance, which then all key alike (cash.lineage_tag).
        if value is not None and taggable(value):
            try:
                value._cash_lineage_hash = hash_
                # This layer re-tags the value whenever it changes, which is
                # what lets the decorator trust the tag for its content.
                value._cash_lineage_src = "statement"
            except (AttributeError, TypeError):
                # Builtins (int / str / ...) and slotted types reject attribute
                # writes. The entry is still authoritative.
                logger.debug("LineageStore: cannot attach _cash_lineage_hash to %r", var)

    def reset_to(self, var: str, hash_: str) -> None:
        """Resynchronise *var*'s lineage to *hash_* without touching the value.

        Used by the simulator when downstream advancement leaves the recorded
        lineage 'ahead' of where simulation says it should be. Distinct from
        :meth:`record` because no fresh computation happened — only state-machine
        correction. ``_cash_lineage_hash`` on the value reflects the value's
        actual computation and must NOT be rewritten here.
        """
        self._lineage[var] = hash_

    def discard(self, var: str) -> None:
        """Forget *var*'s lineage, if it has one."""
        self._lineage.pop(var, None)

    def clear(self) -> None:
        """Forget every lineage."""
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
