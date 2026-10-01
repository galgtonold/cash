"""The library-specific rules of the upstream check, in one place.

The classifier and planner are generic: they compare lineages and walk the
trace. A few library objects need a rule of their own, and each is stated
here so the generic code asks a question instead of naming a library:

* a call on a name that moves its lineage ahead of the simulation without
  making the live value stale (``SELF_CALL_LINEAGE_RESETS``);
* what a matplotlib figure looks like, live or in code
  (``is_figure_carrier``, ``makes_current_figure``, ``figure_save_receiver``).
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Callable, Mapping
from typing import Any

from ...analysis.cacheability_decision import receiver_is_identity_coupled
from ...analysis.namespace_effects import is_estimator
from ..carrier_history import FIGURE_KINDS
from ..stateful_carriers import stateful_carrier_kind

__all__ = [
    "SELF_CALL_LINEAGE_RESETS",
    "bare_receiver_call",
    "figure_save_receiver",
    "is_figure_carrier",
    "makes_current_figure",
    "self_call_resets_lineage",
]


def bare_receiver_call(code: str) -> tuple[str, str] | None:
    """``(receiver, method)`` when *code* is one bare ``name.method(...)``
    expression statement, else ``None``.

    The receiver must be a plain name: ``obj.model.fit(X)`` is a call on
    ``obj.model``, not on a name the caller can look up.
    """
    try:
        body = ast.parse(code.strip()).body
    except (SyntaxError, ValueError):
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Call):
        return None
    func = body[0].value.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return None
    return func.value.id, func.attr


def _quietly(predicate: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """*predicate*, answering no for a value it cannot inspect."""

    def check(value: Any) -> bool:
        try:
            return bool(predicate(value))
        except (TypeError, ValueError, AttributeError):
            return False

    return check


#: ``method -> what the receiver must be`` for a bare ``name.method(...)``
#: that leaves the receiver's live value current while the runtime moves its
#: lineage ahead of the simulation's. When a cell's last change to a required
#: input is such a call and nothing upstream changed, the input's lineage is
#: reset to the simulated one instead of rebuilding the value.
#:
#: * ``fit`` on an estimator: the runtime keys a bare fit on the receiver's
#:   pre-fit lineage and makes the receiver an output, so the fit bumps it.
#:   ``fit`` overwrites the estimator, so a lineage-only reset is safe.
#:   ``partial_fit`` is cumulative and is not listed: resetting only the
#:   lineage of a partially fitted object would count its data twice.
#: * ``savefig`` on a figure: saving counts as a change, so an edit to the
#:   plotted data still redraws and resaves, but it draws nothing, so the live
#:   figure is current. A draw (``ax.plot``) is not listed: re-running it adds
#:   artists, so it must rebuild.
SELF_CALL_LINEAGE_RESETS: Mapping[str, Callable[[Any], bool]] = {
    "fit": _quietly(is_estimator),
    "savefig": _quietly(receiver_is_identity_coupled),
}


def self_call_resets_lineage(var_name: str, last_code: str | None, value: Any) -> bool:
    """Whether *var_name*'s last change, *last_code*, is a call listed in
    ``SELF_CALL_LINEAGE_RESETS`` on *var_name* itself, holding *value*."""
    call = bare_receiver_call(last_code or "")
    if call is None or call[0] != var_name:
        return False
    rule = SELF_CALL_LINEAGE_RESETS.get(call[1])
    return rule is not None and rule(value)


# A pyplot call that REGISTERS a new current figure in the process-global Gcf
# registry -- what ``plt.gcf()`` (and therefore ``plt.savefig()``) resolves to.
_PYPLOT_FIGURE_MAKER = re.compile(r"\b(?:plt|pyplot)\s*\.\s*(?:subplots|subplot_mosaic|figure|subplot|axes)\b")


def is_figure_carrier(value: Any) -> bool:
    """Whether *value* is a live matplotlib Figure or Axes."""
    try:
        return stateful_carrier_kind(value) in FIGURE_KINDS
    except (TypeError, ValueError, AttributeError, RecursionError):
        return False


def makes_current_figure(code: str) -> bool:
    """Whether *code* calls a pyplot function that makes a new current
    figure (``plt.figure()``, ``plt.subplots()``), bound to a name or not."""
    return bool(_PYPLOT_FIGURE_MAKER.search(code))


def figure_save_receiver(code: str) -> str | None:
    """The name *code* calls ``.savefig(...)`` on (``fig.savefig(path)``), or ``None``.

    The module-level ``plt.savefig()`` matches the same shape; telling the two
    apart is the caller's (``statement_saves_current_pyplot_figure``).
    """
    if "savefig" not in code:
        return None
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "savefig"
            and isinstance(node.func.value, ast.Name)
        ):
            return node.func.value.id
    return None
