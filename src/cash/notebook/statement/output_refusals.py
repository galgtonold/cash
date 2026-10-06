"""Output values a statement entry cannot store and restore faithfully.

Checked after execution, because the values do not exist when the
cacheability decision runs, and before the store saves them, because
refusing then is what keeps the RAM tier from deep-copying them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ...analysis.cacheability_decision import identity_coupled_reason
from ..consumables import is_consumable_unrestorable
from ..shared_objects import output_history, shared_names
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


def unrestorable_output_reason(
    outputs: set[str],
    captured_vars: dict[str, Any],
    user_ns: dict[str, Any],
    *,
    cash_held: Iterable[Any] = (),
    shell: Any = None,
) -> str | None:
    """Why one of *outputs* cannot be cached, the first refusal found; None
    when every captured value can. *cash_held* are containers cash itself
    holds (the entries of the cell's call results), whose references do not
    make an output shared.

    ``identity_coupled_reason`` refuses an object identity-coupled to a
    library global: the RAM tier's deep copy of a matplotlib Figure
    re-registers the COPY as pyplot's current figure, so ``plt.savefig()``
    would write the cache's snapshot, a blank PNG on the first run.
    """
    return _value_refusal(outputs, captured_vars, user_ns) or shared_output_reason(
        outputs, captured_vars, user_ns, cash_held, shell
    )


def _value_refusal(outputs: set[str], captured_vars: dict[str, Any], user_ns: dict[str, Any]) -> str | None:
    """The first refusal one output's value earns on its own."""
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


def shared_output_reason(
    outputs: set[str],
    captured_vars: dict[str, Any],
    user_ns: dict[str, Any],
    cash_held: Iterable[Any] = (),
    shell: Any = None,
) -> str | None:
    """Nor an output whose object, or one inside it, something else holds too
    (`shared_names`): a restore would bind a copy, and that holder would keep
    the object the statement really produced or changed -- ``models = {'m': m}``
    restored holds a copy of ``m``, and ``d['a'] = ...`` after ``d = dfs[0]``
    restored leaves ``dfs[0]`` unchanged.

    Asked by reference count, so the frames above must not hold an output's
    value in a local: `unrestorable_output_reason` keeps the per-value loop
    in its own function for that reason. IPython's output history (``Out``,
    ``_``, and *shell*'s copies of them) is not a holder
    (`output_history`)."""
    roots = {out: captured_vars[out] for out in outputs if captured_vars.get(out) is not None}
    history, named = output_history(user_ns, shell)
    shared = shared_names(roots, (captured_vars, user_ns), [*cash_held, *history], named)
    if not shared:
        return None
    names = ", ".join(f"'{n}'" for n in sorted(shared))
    return (
        f"{names} holds an object another variable or container also holds; a restored "
        f"copy would not be that object, so the statement re-runs every time"
    )
