"""What a chart's picture depends on, in a form that survives a kernel restart.

A write provenance record (``_persist_write_provenance``) vouches for a file
by the lineages of what its writer read, and the re-execution planner compares
them with the simulation's after a restart. That works for data. It does not
work for a matplotlib figure: the lineage the runtime gives ``fig`` comes from
facts that exist only in the kernel -- the live Axes -> Figure edges through
which ``ax.bar(...)`` advances ``fig``, and the runtime's verdict on which call
mutated what. After a restart none of that is there, the simulation arrives at
another lineage for the same figure, and every chart writer re-fired with
everything it reads (round 23, r23s2: 84 s to run a table cell after a restart,
re-reading 1,200 exports for two charts nobody asked for).

So for a figure the record carries its **history** instead: the statements
from the one that created it (``fig, ax = plt.subplots()``) to the write, each
with the lineages of what it read -- the data, not the figure. Those lineages
the simulation does reproduce, statement by statement, so the same fingerprint
comes out on both sides exactly when the same picture would be drawn.

Every statement in that span counts, not only the ones recognised as drawing:
missing one (``ax = axes[0]`` and then ``ax.bar(...)``) would let an edit to it
keep a stale chart, while counting one too many only re-draws. For the same
reason there is no fingerprint -- and the writer is judged as before -- when
the span holds anything whose effect a lineage does not show: a control
structure (one unit to the simulation, statement by statement to the runtime),
or a read of a file (``plt.imread(p)``).
"""

from __future__ import annotations

import ast
import hashlib
from typing import Mapping, Sequence

from .cacheability import _READ_TEXT_MARKERS
from .upstream.stateful_carriers import carrier_kind_from_producer

__all__ = ["FIGURE_KINDS", "carrier_history_fingerprint"]

#: The carrier kinds a history stands in for. RNGs are carriers too, but a
#: writer reads what was drawn from one, never the generator itself.
FIGURE_KINDS = frozenset({"matplotlib Figure", "matplotlib Axes"})

_CONTROL = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code.strip())
    except (SyntaxError, ValueError):
        return None


def _assigned_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AnnAssign, ast.AugAssign))
            else []
        )
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _reads_any(tree: ast.Module, names: set[str]) -> bool:
    """Does this statement READ one of *names*?"""
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names for node in ast.walk(tree)
    )


def _counted(tree: ast.Module) -> bool:
    """Does the simulation give this statement a trace entry the way the runtime
    runs it? Bindings and calls do; ``del``, ``pass`` and a bare name shown at
    the end of a cell do not, on either side -- so they are left out of both."""
    return all(
        isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
                *_CONTROL,
            ),
        )
        or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))
        for node in tree.body
    )


def carrier_history_fingerprint(
    statements: Sequence[tuple[str, Mapping[str, str]]],
    carrier: str,
) -> str | None:
    """Fingerprint of the history of figure *carrier* that ends where *statements* end.

    *statements* are the statements that ran before the write, in order and
    from its cell only, each as ``(code, {name: lineage it read})``. Returns
    ``None`` when there is no history to vouch for: *carrier* was last bound by
    something other than a figure-making call (``fig = plt.gcf()``), or the
    span holds a control structure or a file read.
    """
    parsed = [(code, lineages, _parse(code)) for code, lineages in statements]
    parsed = [(code, lin, tree) for code, lin, tree in parsed if tree is not None and _counted(tree)]
    start = None
    for k in range(len(parsed) - 1, -1, -1):
        if carrier in _assigned_names(parsed[k][2]):
            if carrier_kind_from_producer(parsed[k][0]) == "matplotlib Figure":
                start = k
            break
    if start is None:
        return None
    siblings = _assigned_names(parsed[start][2])
    digest = hashlib.sha256()
    for code, lineages, tree in parsed[start:]:
        # What a draw on the figure hands back (``sc = ax.scatter(...)``, then
        # ``fig.colorbar(sc)``) belongs to the figure like ``fig`` and ``ax``:
        # the two sides give it different lineages, so counting it made every
        # chart with a colorbar read as stale (round 30, r30s1 and r30s5). The
        # data that statement read is counted there, where it is read.
        if _reads_any(tree, siblings):
            siblings |= _assigned_names(tree)
        # A loop is hashed as one statement: both the runtime and the
        # simulation log it whole (`StatementProcessor.begin_control_log`).
        # A branch or a `with` still cannot vouch.
        if any(isinstance(node, _CONTROL) and not isinstance(node, _LOOPS) for node in tree.body):
            return None
        if any(marker in code for marker in _READ_TEXT_MARKERS):
            return None
        read = sorted((name, lin) for name, lin in lineages.items() if name not in siblings)
        digest.update(code.strip().encode("utf-8"))
        digest.update(b"\0")
        digest.update(";".join(f"{name}={lin}" for name, lin in read).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
