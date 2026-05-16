"""The single seam for reading and writing variable lineage.

See ``CONTEXT.md`` (entry: *LineageStore*) for the architectural role.
See ``docs/architecture_decisions.md`` ADR-007 for the *cache-key* invariant
this module extends to *lineage state*.

Invariants
----------
* All persistent lineage writes for a variable go through :meth:`record` or
  :meth:`reset_to`. Callers that need to mutate the raw dict directly during
  migration go through :meth:`as_dict` — that surface is transitional.
* When ``record`` is given a ``value`` whose type accepts attributes, the
  dict entry and ``_cash_lineage_hash`` attribute are written together so
  they cannot drift.
* The priority ladder lives in :meth:`resolve`: virtual → store →
  ``value._cash_lineage_hash`` → ``compute_hash_fn`` → ``sha256(str(value))``.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["LineageStore"]


class LineageStore:
    """Owns ``variable_lineage`` and coordinates the paired ``_cash_lineage_hash``
    attribute. See module docstring for invariants."""

    def __init__(self, backing: dict[str, str] | None = None) -> None:
        # Accept an externally-owned dict so existing call sites that hold a
        # reference to ``state.variable_lineage`` keep observing writes during
        # migration.
        self._lineage: dict[str, str] = backing if backing is not None else {}

    # --- read surface ---------------------------------------------------

    def get(self, var: str) -> str | None:
        return self._lineage.get(var)

    def __contains__(self, var: object) -> bool:
        return var in self._lineage

    def __iter__(self) -> Iterator[str]:
        return iter(self._lineage)

    def as_dict(self) -> dict[str, str]:
        """Return the live backing dict. Transitional — prefer :meth:`get` /
        :meth:`record` in new code."""
        return self._lineage

    # --- write surface --------------------------------------------------

    def record(self, var: str, hash_: str, *, value: Any = None) -> None:
        """Record the persistent lineage for *var*.

        When *value* is provided and accepts attributes, also set
        ``value._cash_lineage_hash`` so the dict and the attribute cannot drift.
        """
        self._lineage[var] = hash_
        if value is not None:
            try:
                value._cash_lineage_hash = hash_
            except (AttributeError, TypeError):
                # Builtins (int / str / ...) and slotted types reject attribute
                # writes. The dict is still authoritative.
                logger.debug(
                    "LineageStore: cannot attach _cash_lineage_hash to %r", var
                )

    def reset_to(self, var: str, hash_: str) -> None:
        """Resynchronise *var*'s lineage to *hash_* without touching the value.

        Used by the simulator when downstream advancement leaves the recorded
        lineage 'ahead' of where simulation says it should be. Distinct from
        :meth:`record` because no fresh computation happened — only state-machine
        correction. ``_cash_lineage_hash`` on the value reflects the value's
        actual computation and must NOT be rewritten here.
        """
        self._lineage[var] = hash_

    # --- priority ladder ------------------------------------------------

    def resolve(
        self,
        var: str,
        *,
        value: Any,
        virtual: Mapping[str, str] | None = None,
        compute_hash_fn: Callable[[Any], str] | None = None,
    ) -> str | None:
        """Resolve a lineage hash for *var* using the priority ladder.

        Order: ``virtual`` mapping → store → ``value._cash_lineage_hash`` →
        ``compute_hash_fn(value)`` → ``sha256(str(value))``.
        """
        if virtual is not None and var in virtual:
            return virtual[var]
        if var in self._lineage:
            return self._lineage[var]
        if value is None:
            return None
        try:
            attr = getattr(value, "_cash_lineage_hash", None)
            if attr is not None:
                return attr
            if compute_hash_fn is not None:
                return compute_hash_fn(value)
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        except (AttributeError, TypeError, RecursionError):
            logger.debug("LineageStore: failed to compute lineage for %r", var)
            return None
