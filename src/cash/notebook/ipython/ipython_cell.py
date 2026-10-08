"""A cell holding IPython syntax, as cash runs it: its Python statements
through the pipeline, its magics through IPython, each exactly once.

IPython ran such a cell on its own, so none of its Python was tracked, and the
upstream check, which reads the cell's Python (:meth:`CodeAnalyzer.strip_magics`),
later re-ran those statements in the kernel: a cell's ``print`` or file write
happened twice. cash now runs the cell as IPython's input transform writes it,
where a magic line is a ``get_ipython().run_line_magic(...)`` call
(:func:`~cash.analysis.code_analyzer.calls_ipython`), which runs uncached.

A ``%%time`` or ``%%prun`` cell runs its body through the pipeline from inside
the magic (:class:`CellMagic`), so the magic times or profiles the run cash
made. ``%%capture`` needs no help: it hands its body to ``run_cell``, which is
cash's (the badge is published past the capture, see ``BadgePresenter``).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass

from ...analysis.code_analyzer import CodeAnalyzer, calls_ipython

__all__ = ["CellMagic", "IPythonCell", "ipython_cell"]

#: The cell magics whose body cash runs from inside the magic.
_WRAPPING_MAGICS = frozenset({"time", "prun"})

#: Line magics that load or unload extensions, cash's own among them, or switch
#: cash itself: they re-hook the shell, so not from inside cash's run of a cell.
_SHELL_MAGICS = ("cash", "load_ext", "reload_ext", "unload_ext")


@dataclass(frozen=True)
class CellMagic:
    """The ``%%name line`` a cell's body runs under."""

    name: str
    line: str


@dataclass(frozen=True)
class IPythonCell:
    """A cell's statements as cash runs them.

    *source* is the transformed cell, line for line the user's: a magic line
    is its ``get_ipython()`` call, a cell magic's own line is blank.
    """

    source: str
    tree: ast.Module
    magic: CellMagic | None = None

    def display_text(self, raw_cell: str, node: ast.stmt) -> str | None:
        """The user's own text of a magic *node*, for the badge; None for any
        other statement, whose text in *source* is the user's."""
        if not calls_ipython(node):
            return None
        lines = raw_cell.split("\n")[node.lineno - 1 : node.end_lineno or node.lineno]
        text = "\n".join(lines).strip()
        return text or None


def ipython_cell(raw_cell: str, transform: Callable[[str], str]) -> IPythonCell | None:
    """*raw_cell*, which does not parse as Python, as cash runs it, or None
    when IPython runs it on its own.

    That is a cell magic other than ``%%time`` and ``%%prun``, a cell of
    magic lines and nothing else (nothing to track), a cell that loads an
    extension or switches cash, and one whose transform does not keep its
    lines in place or does not parse (a genuine syntax error).
    """
    lines = raw_cell.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return None
    magic = None
    if lines[first].startswith("%%"):
        name, _, line = lines[first][2:].rstrip().partition(" ")
        if name not in _WRAPPING_MAGICS:
            return None
        magic = CellMagic(name, line)
        lines = [""] * (first + 1) + lines[first + 1 :]
    source = "\n".join(lines)
    if any(line.lstrip().lstrip("%").startswith(_SHELL_MAGICS) for line in lines if line.lstrip().startswith("%")):
        return None
    try:
        return IPythonCell(source, CodeAnalyzer.parse_cell(source), magic)
    except SyntaxError:
        pass
    body = source.lstrip("\n")
    transformed = "\n" * (len(source) - len(body)) + transform(body)
    if transformed.rstrip("\n").count("\n") != source.rstrip("\n").count("\n"):
        return None
    try:
        tree = CodeAnalyzer.parse_cell(transformed)
    except SyntaxError:
        return None
    if magic is None and all(_is_magic_line(node) for node in tree.body):
        return None
    return IPythonCell(transformed, tree, magic)


def _is_magic_line(node: ast.stmt) -> bool:
    """Whether *node* is a magic or a shell command on a line of its own.
    A loop or a branch with one in its body is Python, which cash runs:
    run by IPython, what ``for i in r: %time acc.append(i)`` changed was
    never seen."""
    return isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)) and calls_ipython(node)
