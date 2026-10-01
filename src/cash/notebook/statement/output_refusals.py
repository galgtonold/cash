"""Output values a statement entry cannot store and restore faithfully.

Checked after execution, because the values do not exist when the
cacheability decision runs, and before the store saves them, because
refusing then is what keeps the RAM tier from deep-copying them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...analysis.cacheability_decision import identity_coupled_reason
from ..consumables import is_consumable_unrestorable
from .derivation_edges import is_uncacheable_alias

__all__ = ["unrestorable_output_reason"]


def _alias_refusal(name: str, value: Any, user_ns: dict[str, Any]) -> str | None:
    """A live-alias object (numpy view, pandas groupby/rolling ref-holder)
    is not cached: pickling and restoring it decouples it from its live
    base, so a later base mutation would be lost after restore. It is
    re-derived from the live base instead. ``.copy()`` produces no alias
    and stays cacheable."""
    if is_uncacheable_alias(value, user_ns):
        return f"Live-alias object '{name}' (view/ref-holder); re-derived from live base, not cached."
    return None


def _consumable_refusal(name: str, value: Any) -> str | None:
    """Nor a CONSUMABLE the cache cannot copy -- an open file handle, a
    generator. The RAM tier keeps such a value by reference, so a "hit"
    would hand back the very object a reader already drained: on a second
    Run All `fh = open(p)` would be served, and the cell reading `fh` would
    print [] where Run All in plain Jupyter reads the file again
    (test_a_consumed_iterator_is_rebuilt_for_its_reader)."""
    if is_consumable_unrestorable(value):
        return (
            f"'{name}' is consumed as it is read (an open file or a "
            f"generator) and cannot be restored: it is re-created "
            f"every run"
        )
    return None


def unrestorable_output_reason(outputs: set[str], captured_vars: dict[str, Any], user_ns: dict[str, Any]) -> str | None:
    """Why one of *outputs* cannot be cached, the first refusal found; None
    when every captured value can.

    ``identity_coupled_reason`` refuses an object identity-coupled to a
    library global: the RAM tier's deep copy of a matplotlib Figure
    re-registers the COPY as pyplot's current figure, so ``plt.savefig()``
    would write the cache's snapshot, a blank PNG on the first run.
    """
    refusals: tuple[Callable[[str, Any], str | None], ...] = (
        lambda name, value: _alias_refusal(name, value, user_ns),
        identity_coupled_reason,
        _consumable_refusal,
    )
    for refusal in refusals:
        for out in outputs:
            value = captured_vars.get(out)
            reason = refusal(out, value) if value is not None else None
            if reason is not None:
                return reason
    return None
