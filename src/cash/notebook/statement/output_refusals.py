"""Output values a statement entry cannot store and restore faithfully.

Checked after execution, because the values do not exist when the
cacheability decision runs, and before the store saves them, because
refusing then is what keeps the RAM tier from deep-copying them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from ...analysis.cacheability_decision import identity_coupled_reason
from ..consumables import is_consumable_unrestorable
from ..shared_objects import GaveUp, attrs_copied_apart, output_history, refers_back, share_group, shared_names
from .derivation_edges import is_uncacheable_alias

__all__ = ["is_history_name", "live_shared_reason", "unrestorable_output_reason"]

#: IPython's own names for its input and output history.
_HISTORY_NAMES = re.compile(r"_+|_i+|_\d+|_i\d+|_[iod]h|In|Out")


def is_history_name(name: str) -> bool:
    """Whether *name* is one of IPython's history names (``_``, ``_3``, ``Out``)."""
    return _HISTORY_NAMES.fullmatch(name) is not None


def _alias_refusal(name: str, value: Any, user_ns: dict[str, Any], cash_held: Iterable[Any] = ()) -> str | None:
    """A live-alias object (numpy view, pandas groupby/rolling ref-holder)
    is not cached: pickling and restoring it decouples it from its live
    base, so a later base mutation would be lost after restore. It is
    re-derived from the live base instead. ``.copy()`` produces no alias
    and stays cacheable."""
    if is_uncacheable_alias(value, user_ns, cash_held):
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
    holders: dict[str, Any] | None = None,
    gave_up: list[GaveUp] | None = None,
) -> str | None:
    """Why one of *outputs* cannot be cached, the first refusal found; None
    when every captured value can. *cash_held* are containers cash itself
    holds (the entries of the cell's call results), whose references do not
    make an output shared.

    With *holders*, a variable of the notebook that holds an output's object
    too does not refuse the statement: it is added to *holders*, to be
    stored and restored with the outputs (`shared_output_reason`), and its
    value must pass the same checks. *gave_up* gets a `GaveUp` appended when
    the check for those holders ran and found a holder that is not a
    variable: the variables it found holding the outputs' objects all the
    same, to link though the statement is not stored.

    ``identity_coupled_reason`` refuses an object identity-coupled to a
    library global: the RAM tier's deep copy of a matplotlib Figure
    re-registers the COPY as pyplot's current figure, so ``plt.savefig()``
    would write the cache's snapshot, a blank PNG on the first run.
    """
    cash_held = list(cash_held)
    reason = _value_refusal(outputs, captured_vars, user_ns, cash_held) or shared_output_reason(
        outputs, captured_vars, user_ns, cash_held, shell, holders, gave_up
    )
    if reason is None and holders:
        reason = _value_refusal(set(holders), holders, user_ns, cash_held)
        if reason is not None:
            holders.clear()
    if reason is None and _attrs_parted(outputs, captured_vars, holders):
        reason = (
            "a pandas object's .attrs holds an object that the statement's other values hold too; "
            "pandas copies .attrs apart on restore, so the statement re-runs every time"
        )
        if holders:
            holders.clear()
    return reason


def _attrs_parted(outputs: set[str], captured_vars: dict[str, Any], holders: dict[str, Any] | None) -> bool:
    """`attrs_copied_apart` for the values the entry would store together."""
    roots = [captured_vars[out] for out in outputs if captured_vars.get(out) is not None]
    return attrs_copied_apart([*roots, *(holders or {}).values()])


def _value_refusal(
    outputs: set[str], captured_vars: dict[str, Any], user_ns: dict[str, Any], cash_held: Iterable[Any] = ()
) -> str | None:
    """The first refusal one output's value earns on its own."""
    refusals: tuple[Callable[[str, Any], str | None], ...] = (
        lambda name, value: _alias_refusal(name, value, user_ns, cash_held),
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
    holders: dict[str, Any] | None = None,
    gave_up: list[GaveUp] | None = None,
) -> str | None:
    """Nor an output whose object, or one inside it, something else holds too
    (`shared_names`): a restore would bind a copy, and that holder would keep
    the object the statement really produced or changed -- ``models = {'m': m}``
    restored holds a copy of ``m``, and ``d['a'] = ...`` after ``d = dfs[0]``
    restored leaves ``dfs[0]`` unchanged.

    Unless the holders are variables of the notebook, directly or through the
    containers in them, and the caller passed *holders* to take them
    (`share_group`): stored and restored with the outputs as one graph, they
    hold the restored objects again. A holder cash cannot restore -- a
    library's registry, a closure, IPython's own names -- still refuses.

    Asked by reference count, so the frames above must not hold an output's
    value in a local: `unrestorable_output_reason` keeps the per-value loop
    in its own function for that reason. IPython's output history (``Out``,
    ``_``, and *shell*'s copies of them) is not a holder
    (`output_history`)."""
    history, named = output_history(user_ns, shell)
    names = [out for out in outputs if captured_vars.get(out) is not None]
    found_holders: set[str] | None = None
    if holders is None:
        roots = {out: captured_vars[out] for out in names}
        shared = shared_names(roots, (captured_vars, user_ns), [*cash_held, *history], named)
        del roots
    else:
        hidden = getattr(shell, "user_ns_hidden", None) or {}
        partial = GaveUp()
        found, shared = share_group(
            names,
            captured_vars,
            user_ns,
            [*cash_held, *history],
            named,
            foreign=lambda name: name in hidden or is_history_name(name),
            gave_up=partial,
        )
        holders.update(found)
        del found
        found_holders = partial.names
        if shared and gave_up is not None:
            gave_up.append(partial)
    return _shared_reason(shared, captured_vars, found_holders)


def _shared_reason(
    shared: set[str], captured_vars: dict[str, Any], found_holders: set[str] | None = None
) -> str | None:
    """The refusal for the outputs *shared*. When no variable was found
    holding them (*found_holders*, when the search ran) and an output's
    object is held by its own parts (`refers_back`: each sheet of an
    openpyxl workbook holds the workbook), the count cannot tell those
    references from another holder's, and the reason says that instead of
    naming a holder that is not there."""
    if not shared:
        return None
    selfheld = set()
    if not found_holders:
        selfheld = {name for name in shared if refers_back(captured_vars.get(name))}
    reasons = []
    others = shared - selfheld
    if others:
        names = ", ".join(f"'{n}'" for n in sorted(others))
        reasons.append(f"{names} holds an object another variable or container also holds")
    for name in sorted(selfheld):
        kind = type(captured_vars.get(name)).__name__
        reasons.append(
            f"'{name}' is a {kind} whose own parts hold it, so cash cannot tell whether anything else holds it too"
        )
    return "; ".join(reasons) + "; a restored copy would not be that object, so the statement re-runs every time"


def live_shared_reason(
    names: Iterable[str],
    keep_identity: Iterable[str],
    user_ns: dict[str, Any],
    cash_held: Iterable[Any] = (),
    shell: Any = None,
) -> str | None:
    """Why a hit may not restore over the objects *names* are bound to now.

    A hit binds copies. For the names in *keep_identity* -- what the
    statement changes in place, and the variables its entry stores with its
    outputs -- the run would have kept the very object, so every other
    holder of it would have seen the change. When something outside *names*
    holds one of those objects now (``b = a`` run after the entry was
    stored), the statement runs instead, and stores the new group.
    """
    history, named = output_history(user_ns, shell)
    roots = {name: user_ns[name] for name in names if user_ns.get(name) is not None}
    shared = shared_names(roots, (user_ns,), [*cash_held, *history], named, keep_identity=keep_identity)
    del roots
    if not shared:
        return None
    listed = ", ".join(f"'{n}'" for n in sorted(shared))
    return f"another variable or container now holds the object {listed} is bound to; the statement runs again"
