"""Which trace statements belong to a stateful carrier's history.

A carrier (a figure, a seeded RNG; see ``stateful_carriers``) is filled by
statements whose recorded outputs may not name it: a loop drawing on ``ax``,
a call handed ``ax=axes[0]``. The planner's carrier-history pass and the
figure-write guard decide "is this statement a fill" with the one predicate
here, :func:`fills_carrier`.
"""

from __future__ import annotations

import ast
import textwrap

__all__ = ["fills_carrier", "passes_carrier_to_a_call"]


_CONTROL_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
)


def _control_body_touches(code: str, sibling_names: set[str]) -> bool:
    """True when a CONTROL STRUCTURE's body calls a method on one of *sibling_names*.

     The simulation treats a loop / ``if`` / ``with`` as ONE trace entry and does
     not surface the mutations performed inside its body, so the statement's
     recorded outputs never mention ``ax`` for::

         for c in summary.columns:
             ax.plot(range(len(summary)), summary[c].values, label=c)

     :meth:`ReexecutionPlanner._complete_stateful_carrier_history` keys on exactly
     those outputs, so the loop was left out of the plan while
     ``fig, ax = plt.subplots()`` and ``fig.savefig(path)`` were scheduled -- the
     figure was rebuilt EMPTY and the blank PNG was written over the good chart
    . That method's docstring already describes this failure for the
     flat ``ax.bar(...)`` form; only the loop shape escaped, because the flat one
     IS visible in the outputs.

     Deliberately restricted to control structures: the flat form is already
     covered by the outputs check, and widening this to plain statements would
     also promote pure reads (``ax.get_title()``), risking exactly the
     over-scheduling regressions that method is documented to be narrow about.
    """
    if not sibling_names:
        return False
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError, TypeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _CONTROL_NODES):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id in sibling_names
            ):
                return True
    return False


def _root_name(node: ast.AST) -> str | None:
    """``axes[0]`` / ``ax.twinx()`` / ``*axs`` -> the name they are reached from."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def passes_carrier_to_a_call(code: str, sibling_names: set[str]) -> bool:
    """True when the statement calls into one of *sibling_names* or hands it to a call.

    Two ways to draw on a figure that its recorded outputs do not show:

    * a call on a PART of it -- ``axes[1].set_xlabel(...)``,
      ``ax.xaxis.set_major_formatter(...)``: the receiver is reached from the
      carrier, but is not a bare name;
    * a call on SOMETHING ELSE that receives it -- ``tot.plot(ax=axes[0])``,
      ``imp.plot.barh(..., ax=ax)``, ``sns.barplot(data=df, ax=ax)``,
      ``draw_panel(ax)``. The outputs name the receiver (``tot``, ``imp``),
      never ``ax``.

    Missing either re-drew the figure without it, as a blank chart. A call
    that merely
    reads the axes is re-run too: within the carrier's own history, one
    statement too many is harmless; one too few writes a wrong file.
    """
    if not sibling_names:
        return False
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError, TypeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _root_name(node.func) in sibling_names:
            return True
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if _root_name(arg) in sibling_names:
                return True
    return False


def fills_carrier(entry, sibling_names: set[str]) -> bool:
    """Is trace *entry* part of the history of a carrier co-produced as *sibling_names*?

    One predicate for both the pass that completes a carrier's history and the
    guard that refuses a write whose history is incomplete -- when they
    disagreed, the guard let a blank chart through that the pass never saw.
    """
    code = entry.stmt_code
    return bool(
        set(entry.outputs) & sibling_names
        or _control_body_touches(code, sibling_names)
        or passes_carrier_to_a_call(code, sibling_names)
    )
