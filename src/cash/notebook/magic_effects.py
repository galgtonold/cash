"""What a line magic or a shell command binds, as the runtime and the upstream
simulation both record it.

A magic runs every time, uncached, and IPython, not code cash can read, binds
what it binds. The names it binds or changes -- the target of ``x = !cmd`` or
``res["k"] = %sx cmd``, what ``%time df = clean(df)`` assigns, the receiver of
``%time model.fit()`` -- get a lineage of their own: the magic's text, the
lineage of every name it reads, and a digest of the value it left
(:func:`magic_output_lineage`). The runtime records it when the magic runs
and the simulation reaches the same lineage from the digests the run left
(``TrackingState.magic_values``), so a cell below finds them as it left them.

Such a lineage marks a value the upstream check never rebuilds from the Python
above it: that would drop what the magic did to it. When the simulation's
lineage differs from the live one, something the magic read changed since it
ran, and the check warns instead (``NOTEBOOK-MAGIC-STALE``).
"""

from __future__ import annotations

import ast
import functools
import hashlib
from collections.abc import Callable, Mapping

from ..analysis.code_analyzer import CodeAnalyzer, calls_ipython
from ..analysis.mutations import KNOWN_PURE_METHODS

__all__ = [
    "is_magic_statement",
    "magic_base",
    "magic_cell_of",
    "magic_effects",
    "magic_output_lineage",
    "simulation_cell",
]

#: Line magics whose argument is a Python statement run in the user's namespace.
_PYTHON_ARG_MAGICS = frozenset({"time", "timeit", "prun"})


def is_magic_statement(node: ast.stmt) -> bool:
    """Whether *node*, a statement as IPython's transform writes it, runs a magic
    or a shell command itself (``get_ipython().run_line_magic(...)``,
    ``x = get_ipython().getoutput(...)``), not inside a block or a definition."""
    return isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)) and calls_ipython(node)


def _python_argument(arg: str) -> ast.Module | None:
    """The statement a ``%time``/``%timeit``/``%prun`` line runs: *arg* without
    the options before it."""
    words = arg.split(" ")
    for start in range(len(words)):
        try:
            return CodeAnalyzer.parse_cell(" ".join(words[start:]).strip())
        except SyntaxError:
            continue
    return None


def _changed_receivers(tree: ast.Module) -> set[str]:
    """Names a method call in *tree* may change in place: the receiver of every
    method not known to leave it alone (``model.fit()``, not ``df.head()``)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in KNOWN_PURE_METHODS:
                continue
            base = node.func.value
            while isinstance(base, (ast.Attribute, ast.Subscript, ast.Call)):
                base = base.func if isinstance(base, ast.Call) else base.value
            if isinstance(base, ast.Name):
                names.add(base.id)
    return names


def magic_effects(node: ast.stmt, is_module: Callable[[str], bool]) -> tuple[set[str], set[str]]:
    """``(changed, read)``: the names the magic statement *node* binds or may
    change, and every name it reads (the changed ones among them).

    The statement's own targets (``files = !ls``), and, for a line magic that
    runs a Python statement (``%time``, ``%timeit``, ``%prun``), what that
    statement assigns and the receivers its method calls may change. A module
    (*is_module*) is never counted as changed: its lineage is its source.
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(ast.unparse(node))
    changed = set(outputs)
    read = set(inputs) - {"get_ipython"}
    for call in ast.walk(node):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "run_line_magic"
            and len(call.args) == 2
            and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in call.args)
            and call.args[0].value in _PYTHON_ARG_MAGICS
        ):
            continue
        inner = _python_argument(call.args[1].value)
        if inner is None:
            continue
        inner_inputs, inner_outputs = CodeAnalyzer.analyze_code_block(ast.unparse(inner))
        changed |= set(inner_outputs) | _changed_receivers(inner)
        read |= set(inner_inputs)
    changed = {name for name in changed if not is_module(name)}
    return changed, read | changed


def magic_base(code: str, read_lineages: Mapping[str, str | None]) -> str:
    """What the magic statement *code* depends on: its text and the lineage of
    each name it reads (none for a name without one)."""
    reads = ";".join(f"{name}={read_lineages[name] or ''}" for name in sorted(read_lineages))
    return hashlib.sha256(f"magic:{code}:{reads}".encode("utf-8")).hexdigest()


def magic_output_lineage(base: str, digest: str) -> str:
    """The lineage of a name the magic with *base* left holding a value with *digest*."""
    return hashlib.sha256(f"{base}:value:{digest}".encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=512)
def simulation_cell(cell_code: str) -> tuple[str, ast.Module] | None:
    """*cell_code* as the runtime runs it when it holds IPython syntax: the
    transformed source and its tree (``ipython_cell``), or None for a cell of
    Python or one IPython runs on its own.

    A cell of magics and nothing else, which IPython runs on its own, is read
    the same way: the magics in it bind what they bind either way.
    """
    try:
        CodeAnalyzer.parse_cell(cell_code)
        return None
    except SyntaxError:
        pass
    from IPython.core.inputtransformer2 import TransformerManager

    from .ipython.ipython_cell import ipython_cell

    transform = TransformerManager().transform_cell
    cell = ipython_cell(cell_code, transform)
    if cell is not None:
        return cell.source, cell.tree
    return _magics_only(cell_code, transform)


def _magics_only(cell_code: str, transform: Callable[[str], str]) -> tuple[str, ast.Module] | None:
    """A cell of line magics and shell commands only, transformed, or None."""
    lines = cell_code.split("\n")
    first = next((line for line in lines if line.strip()), "")
    if first.startswith("%%"):
        return None
    body = cell_code.lstrip("\n")
    transformed = "\n" * (len(cell_code) - len(body)) + transform(body)
    if transformed.rstrip("\n").count("\n") != cell_code.rstrip("\n").count("\n"):
        return None
    try:
        tree = CodeAnalyzer.parse_cell(transformed)
    except SyntaxError:
        return None
    if not tree.body or not all(is_magic_statement(node) for node in tree.body):
        return None
    return transformed, tree


def magic_cell_of(name: str, cells: list[str]) -> tuple[int, str] | None:
    """``(index, line)``: the last of *cells* with a magic that binds or may
    change *name*, and that magic's line as the user wrote it; None if none does."""
    for idx in range(len(cells) - 1, -1, -1):
        cell = simulation_cell(cells[idx].replace("\r\n", "\n"))
        if cell is None:
            continue
        lines = cells[idx].replace("\r\n", "\n").split("\n")
        for node in reversed(cell[1].body):
            if is_magic_statement(node) and name in magic_effects(node, lambda _name: False)[0]:
                return idx, lines[node.lineno - 1].strip()
    return None
