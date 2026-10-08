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
above it alone: that would drop what the magic did to it. A ``%time``,
``%timeit`` or ``%prun`` line runs Python (:func:`is_rerun_magic`), so a
rebuild runs it again after that Python, as a top-to-bottom run does. For any
other magic, when the simulation's lineage differs from the live one,
something the magic read changed since it ran, and the check warns instead
(``NOTEBOOK-MAGIC-STALE``).
"""

from __future__ import annotations

import ast
import functools
import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from ..analysis.code_analyzer import CodeAnalyzer, calls_ipython
from ..analysis.code_analyzer import python_magic_argument as _python_magic_arg
from ..analysis.code_analyzer import split_magic_argument as _split_argument
from ..analysis.mutations import KNOWN_PURE_METHODS
from ..tracking.randomness import advanced_rng_lineage, get_drawing_rng_modules, rng_virtual_var
from ..analysis.namespace_effects import bare_call_arguments, call_arguments

__all__ = [
    "is_magic_statement",
    "is_rerun_magic",
    "magic_base",
    "magic_call_arguments",
    "magic_cell_of",
    "magic_effects",
    "magic_output_lineage",
    "simulation_cell",
]


def is_magic_statement(node: ast.stmt) -> bool:
    """Whether *node*, a statement as IPython's transform writes it, runs a magic
    or a shell command itself (``get_ipython().run_line_magic(...)``,
    ``x = get_ipython().getoutput(...)``), not inside a block or a definition."""
    return isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)) and calls_ipython(node)


def is_rerun_magic(node: ast.stmt) -> bool:
    """Whether the magic statement *node* runs nothing but Python cash can
    read: ``%time``, ``%timeit`` and ``%prun`` lines whose statement is
    Python with no magic or shell command of its own.

    Such a statement does what its Python does, so a rebuild runs it again,
    as a top-to-bottom run does. Any other magic or shell command is never
    run for the user (``NOTEBOOK-MAGIC-STALE``).
    """
    found = False
    for call in ast.walk(node):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Name)
            and call.func.value.func.id == "get_ipython"
        ):
            continue
        arg = _python_magic_arg(call)
        split = None if arg is None else _split_argument(arg)
        # What comes before the statement must be options: `%time %time f()`
        # parses as `f()` once its first word is dropped.
        if split is None or calls_ipython(split[1]) or any(not w.startswith("-") for w in split[0][:1]):
            return False
        if any(w.startswith(("%", "!")) for w in split[0]):
            return False
        found = True
    return found


def _python_argument(arg: str) -> ast.Module | None:
    """The statement a ``%time``/``%timeit``/``%prun`` line runs: *arg* without
    the options before it."""
    split = _split_argument(arg)
    return None if split is None else split[1]


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
    change, and every name it reads (among them the changed ones it does not
    only bind, see `_only_bound`).

    The statement's own targets (``files = !ls``), and, for a line magic that
    runs a Python statement (``%time``, ``%timeit``, ``%prun``), what that
    statement assigns and the receivers its method calls may change. A module
    (*is_module*) is never counted as changed: its lineage is its source.
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(ast.unparse(node))
    changed = set(outputs)
    read = set(inputs) - {"get_ipython"}
    trees: list[ast.AST] = [node]
    for call in ast.walk(node):
        arg = _python_magic_arg(call)
        if arg is None:
            continue
        inner = _python_argument(arg)
        if inner is None:
            continue
        trees.append(inner)
        inner_inputs, inner_outputs = CodeAnalyzer.analyze_code_block(ast.unparse(inner))
        changed |= set(inner_outputs) | _changed_receivers(inner)
        read |= set(inner_inputs)
    changed = {name for name in changed if not is_module(name)}
    return changed, read | (changed - _only_bound(trees))


def _only_bound(trees: list[ast.AST]) -> set[str]:
    """The names *trees* only bind as a whole (``x = ...``, ``for x in``), never
    read: the value the magic gives such a name does not depend on the one it
    had. Counted as read, its lineage before the magic went into the magic's
    base: none at run time on a first run, the one the run left in the
    simulation, so the two never matched and the reader below re-ran the
    magic (or warned it stale) on every Run All."""
    stored: set[str] = set()
    loaded: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign):
                loaded.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.Name):
                (loaded if isinstance(node.ctx, ast.Load) else stored).add(node.id)
    return stored - loaded


def magic_rng_advances(node: ast.stmt, code: str, lineage: Mapping[str, str]) -> dict[str, str]:
    """The lineage each seeded RNG variable takes after the magic statement
    *node* (*code*) ran a draw (``%time a = np.random.rand(2)``), from the
    variable's *lineage* before it: a draw moves the stream on, as a
    statement's does (``advanced_rng_lineage``). Both engines call this."""
    advances: dict[str, str] = {}
    for call in ast.walk(node):
        arg = _python_magic_arg(call)
        inner = _python_argument(arg) if arg is not None else None
        if inner is None:
            continue
        try:
            modules = get_drawing_rng_modules(ast.unparse(inner))
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            continue
        for module in modules:
            var = rng_virtual_var(module)
            before = advances.get(var, lineage.get(var))
            if before:
                advances[var] = advanced_rng_lineage(magic_base(code, {var: before}), var)
    return advances


def magic_call_arguments(node: ast.stmt, user_ns: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """``(watched, bare)``: the names the Python statement of a ``%time``,
    ``%timeit`` or ``%prun`` line in *node* hands to a call that could change
    them in place, as a plain statement's are watched
    (``namespace_effects.call_arguments``), and those among them a bare call
    is handed.

    ``%time train(model)`` changes ``model`` as much as ``train(model)``
    does, but no method is called on it, so it is no receiver
    (:func:`magic_effects`). The runtime fingerprints these around the magic
    and records the ones it changed (``MutationClassifier.magic_snapshots``).
    """
    watched: set[str] = set()
    bare: set[str] = set()
    for call in ast.walk(node):
        arg = _python_magic_arg(call)
        inner = None if arg is None else _python_argument(arg)
        if inner is None:
            continue
        watched |= call_arguments(inner, user_ns)
        bare |= bare_call_arguments(inner, user_ns)
    return frozenset(watched), frozenset(bare & watched)


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
