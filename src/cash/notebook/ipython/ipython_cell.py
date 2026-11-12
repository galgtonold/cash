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

from ...analysis.code_analyzer import CodeAnalyzer, calls_ipython, parse_cell_source

__all__ = ["CellMagic", "IPythonCell", "ipython_cell", "runs_no_user_code"]

#: The cell magics whose body cash runs from inside the magic.
_WRAPPING_MAGICS = frozenset({"time", "prun"})

#: Line magics that switch cash itself: they re-hook the shell, so not from
#: inside cash's run of a cell. Cash's other magics (``%cash_badge print``,
#: ``%cash_debug on``) only change a setting and run in place: left to IPython,
#: they took the whole cell, so a setup cell's ``COLS = [...]`` ran untracked
#: on every Run All, and what was keyed on it missed.
_SHELL_MAGICS = frozenset({"cash_on", "cash_off"})

#: Line magics that load or unload an extension: IPython's to run when the
#: extension is cash's own (`_rehooks_cash`).
_EXTENSION_MAGICS = frozenset({"load_ext", "reload_ext", "unload_ext"})


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
    magic lines and nothing else (nothing to track), a cell that switches
    cash or loads, reloads or unloads its extension, and one whose transform does not keep its
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
    if any(_rehooks_cash(line) for line in lines):
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


#: Line magics that run none of the user's Python and bind or change no name
#: in the namespace: packages, the working directory, the environment and
#: listings. ``%run``, ``%time``, ``%store -r``, ``%reset``, ``%load_ext``
#: run code or change names; ``%whos`` and ``%pinfo`` call a value's own
#: ``repr`` or attributes. They stay off this list.
_INERT_LINE_MAGICS = frozenset(
    {
        "pip",
        "conda",
        "mamba",
        "micromamba",
        "uv",
        "pwd",
        "cd",
        "env",
        "ls",
        "ll",
        "who",
        "who_ls",
        "lsmagic",
        "magic",
        "history",
        "hist",
    }
)


def runs_no_user_code(raw_cell: str, transform: Callable[[str], str]) -> bool:
    """Whether IPython, running *raw_cell*, provably runs none of the user's
    Python and binds no name, for a cell the upstream check cannot read.

    That is a cell whose text, after IPython's own input transform
    (*transform*: magics, automagic, shell escapes), does not parse: IPython
    shows the SyntaxError before it runs a statement. Or a cell that is a
    single line magic written without its ``%`` (automagic: ``pip install
    x``) from :data:`_INERT_LINE_MAGICS`, which IPython runs just as it runs
    the same line with its ``%``.

    A cell the upstream check reads (``%cd x`` with its ``%`` included) is
    not this function's: False, as for anything else not proved here.
    """
    if parse_cell_source(raw_cell) is not None:
        return False
    try:
        transformed = transform(raw_cell)
    except SyntaxError:
        return True  # IPython shows it and runs nothing
    except Exception:  # noqa: BLE001 - not proved; the caller takes its usual path
        return False
    try:
        tree = CodeAnalyzer.parse_cell(transformed)
    except SyntaxError:
        return True
    except Exception:  # noqa: BLE001 - a null byte: not proved
        return False
    return bool(tree.body) and all(_is_inert_magic(node) for node in tree.body)


def _is_inert_magic(node: ast.stmt) -> bool:
    """Whether *node* is ``get_ipython().run_line_magic('<name>', '<line>')``
    on its own, *name* one of :data:`_INERT_LINE_MAGICS` and *line* naming no
    Python to expand."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run_line_magic"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "get_ipython"
        and len(call.args) == 2
        and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in call.args)
    ):
        return False
    name, line = (arg.value for arg in call.args)  # type: ignore[attr-defined]
    # IPython expands ``{expr}`` and ``$name`` in the line before the magic
    # sees it (``var_expand``), evaluating the user's Python.
    return name in _INERT_LINE_MAGICS and "{" not in line and "$" not in line


def _rehooks_cash(line: str) -> bool:
    """Whether the magic *line* switches cash (``%cash_off``) or loads,
    reloads or unloads an extension of cash's own (``%load_ext cash``).

    Another extension's ``%load_ext`` (``autoreload``) runs in place as the
    other magics do. Left to IPython, it took the whole cell, so the imports
    beside it (``%load_ext autoreload`` / ``%autoreload 2`` / ``from sklearn
    import ...``, the usual first cell) ran outside the pipeline, the names
    they bound had no lineage, and every statement reading one ran uncached.
    """
    stripped = line.lstrip()
    if not stripped.startswith("%"):
        return False
    name, _, rest = stripped.lstrip("%").partition(" ")
    if name in _SHELL_MAGICS:
        return True
    if name not in _EXTENSION_MAGICS:
        return False
    return any(ext == "cash" or ext.startswith("cash.") for ext in rest.split())


def _is_magic_line(node: ast.stmt) -> bool:
    """Whether *node* is a magic or a shell command on a line of its own.
    A loop or a branch with one in its body is Python, which cash runs:
    run by IPython, what ``for i in r: %time acc.append(i)`` changed was
    never seen."""
    return isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)) and calls_ipython(node)
