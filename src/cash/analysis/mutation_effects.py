"""What running a cell or a statement changes, as one typed result.

The upstream checker needs to know everything a cell writes, split by the
channel an isolated re-run resets it through (:class:`CellEffects`); the
statement processor and the upstream simulator need to know what one statement
reads and writes (:class:`StatementEffects`). Each used to chain the analyses
in the statement analyses by hand. They now ask :func:`cell_effects` and
:func:`statement_effects`, so a new mutation channel is added in one place.

Functions called by name are resolved through a ``name -> source`` callable
the caller supplies: the runtime reads the live namespace, the simulation and
the checker read the notebook's cell text. Only the resolver differs between
engines; the rules do not.
"""

from __future__ import annotations

import ast
import functools
import inspect
import os
import pathlib
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..content_hashers import builtin_hash_family
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..tracking.randomness.state import rng_carrier_kind
from ..value_types import BUILTIN_NAMES
from .aliases import aliased_sources
from .analysis_budget import ANALYSIS_BUDGET_REASON, AnalysisBudgetExceeded, no_analysis_budget
from .annotations import extract_annotations_for_statements
from .ast_util import called_names
from .cacheability import analyze_statement
from .cacheability_decision import analysis_failed, receiver_is_identity_coupled
from .callee_effects import (
    all_param_names,
    argument_roots,
    callee_global_mutations,
    function_arg_mutations,
    handed_or_called_on,
    iterated_sources,
    mutating_partials,
    partial_arg_mutations,
    reduce_free_mutations,
    resolve_function_def,
    stateful_closure_vars,
    stateful_self_functions,
)
from .code_analyzer import CodeAnalyzer, magic_python, parse_cell_source
from .handed_callables import handed_callables, one_reading
from .mutations import (
    MUTATING_METHODS,
    RECEIVER_READONLY_WRITE_METHODS,
    TopLevelCalls,
    assigned_method_call_receivers,
    chain_is_pure,
    crossref_reassigned_vars,
    is_pandas_plot_call,
    selfref_inplace_write_vars,
    selfref_reassignment_targets,
    standalone_method_mutation_receivers,
    subscript_view_bindings,
    top_level_calls,
)
from .namespace_effects import (
    ARGUMENT_READING_BUILTINS,
    call_only_reads,
    calls_with_lambdas,
    capturable_globals,
    fits_its_receiver,
    function_handoffs,
    handed_to_params,
    notebook_global_rebinds,
    users_callee,
)
from .object_protocol import ObjectProtocolResets, changed_through_their_objects, object_protocol_mutations

__all__ = [
    "CellEffects",
    "NotebookSources",
    "ReceiverClasses",
    "StatementEffects",
    "cell_effects",
    "captured_call_receiver_names",
    "captured_call_receivers",
    "classify_receivers",
    "comprehension_watch_candidates",
    "condition_mutations",
    "control_structure_mutations",
    "drawn_on_arguments",
    "is_module_name",
    "live_function_source",
    "loop_target_roots",
    "loop_watch_candidates",
    "WATCHED_STRUCTURES",
    "statement_watch_candidates",
    "statement_effects",
]

SourceResolver = Callable[[str], "str | None"]


def live_function_source(name: str, namespace: Mapping[str, Any]) -> str | None:
    """Source of the function *name* as the kernel holds it, or None.

    *name* bound in *namespace*, else a module-level helper in the globals of
    any function there: an imported ``build(ds)`` calling ``clean(ds)``, with
    ``clean`` living in ``build``'s module and never in the notebook, resolves
    too. The runtime resolves through this; the simulation falls back to it
    for what the cell text does not define.
    """
    fn = namespace.get(name)
    if callable(fn) and not isinstance(fn, type):
        try:
            return inspect.getsource(fn)
        except SOURCE_RETRIEVAL_ERRORS:
            pass
    if name not in namespace and name in BUILTIN_NAMES:
        # ``len`` or ``print`` unbound in the namespace is the builtin, which
        # has no source. Scanning every value's globals for it took about 1 ms
        # per call with 3,000 functions in the namespace.
        return None
    seen: set[int] = set()
    for value in namespace.values():
        module_globals = getattr(value, "__globals__", None)
        if not isinstance(module_globals, dict) or id(module_globals) in seen:
            continue
        seen.add(id(module_globals))
        candidate = module_globals.get(name)
        if callable(candidate) and not isinstance(candidate, type):
            try:
                return inspect.getsource(candidate)
            except SOURCE_RETRIEVAL_ERRORS:
                continue
    return None


# ---------------------------------------------------------------------------
# Notebook-wide definitions, read from cell text
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8192)
def cell_definitions(code: str, kinds: type | tuple[type, ...]) -> tuple[tuple[str, str], ...]:
    """``(name, source)`` of each top-level definition of *kinds* in the cell
    *code*, in order. One cell's text decides it, and the upstream check asks
    for every cell's on every cell: unparsing them all again was 0.3 s of the
    last 20 cells of a 400-cell notebook."""
    tree = parse_cell_source(code)
    found = []
    for node in tree.body if tree is not None else ():
        if isinstance(node, kinds):
            try:
                found.append((node.name, ast.unparse(node)))
            except (ValueError, AttributeError):
                continue
    return tuple(found)


class NotebookSources:
    """Top-level definitions across a notebook's cells plus the cell being run.

    Read from cell SOURCE, not ``inspect.getsource``: under nbclient a
    cell-defined function or class has no ``linecache`` entry. Where a name is
    defined or bound more than once, the last one wins, as in a top-to-bottom
    run. *cells* is called at most once, on first use, so a cell that needs no
    notebook-wide lookup never reads the notebook.
    """

    def __init__(self, cells: Callable[[], list[str]], current_cell: str) -> None:
        self._load_cells = cells
        self._current_cell = current_cell

    @functools.cached_property
    def cells(self) -> list[str]:
        """The notebook's cell sources (the current cell among them, when saved)."""
        return list(self._load_cells())

    @functools.cached_property
    def _bodies(self) -> list[list[ast.stmt]]:
        # Magics stripped: ``%matplotlib inline`` above a ``def`` is not a
        # reason to lose the ``def``.
        trees = (parse_cell_source(code) for code in (*self.cells, self._current_cell))
        return [tree.body for tree in trees if tree is not None]

    def _top_level(self, kinds) -> Iterable:
        for body in self._bodies:
            for node in body:
                if isinstance(node, kinds):
                    yield node

    @functools.cached_property
    def functions(self) -> dict[str, str]:
        """``{function_name: source}`` for every top-level ``def``."""
        return self._unparsed((ast.FunctionDef, ast.AsyncFunctionDef))

    @functools.cached_property
    def classes(self) -> dict[str, str]:
        """``{class_name: source}`` for every top-level ``class``."""
        return self._unparsed(ast.ClassDef)

    def _unparsed(self, kinds) -> dict[str, str]:
        sources: dict[str, str] = {}
        for code in (*self.cells, self._current_cell):
            sources.update(cell_definitions(code, kinds))
        return sources

    @functools.cached_property
    def decorated_instances(self) -> dict[str, str]:
        """``{func_name: decorator_name}`` for each top-level ``@Deco def f``.

        When the decorator is a class, ``f`` is an INSTANCE of it
        (``@Counter def task``). Only the first bare-``Name`` decorator counts.
        """
        instances: dict[str, str] = {}
        for node in self._top_level((ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    instances[node.name] = dec.id
                    break
        return instances

    @functools.cached_property
    def name_aliases(self) -> dict[str, str]:
        """``{alias: source}`` for each top-level ``alias = source`` (``h = g``)."""
        return {
            node.targets[0].id: node.value.id
            for node in self._top_level(ast.Assign)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name)
        }

    @functools.cached_property
    def var_factories(self) -> dict[str, str]:
        """``{var: factory_name}`` for each top-level ``var = factory(...)``."""
        return {
            node.targets[0].id: node.value.func.id
            for node in self._top_level(ast.Assign)
            if len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        }

    @functools.cached_property
    def partial_bindings(self) -> dict[str, tuple[str, list]]:
        """``{var: (target_func, [bound_arg_vars])}`` for each top-level
        ``var = partial(f, x, ...)`` / ``functools.partial(...)``. Bound args
        are Name ids, or ``None`` for anything else, aligned to ``f``'s
        positional parameters."""
        bindings: dict[str, tuple[str, list]] = {}
        for node in self._top_level(ast.Assign):
            if not (
                len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
            ):
                continue
            call = node.value
            func = call.func
            is_partial = (isinstance(func, ast.Name) and func.id == "partial") or (
                isinstance(func, ast.Attribute) and func.attr == "partial"
            )
            if is_partial and call.args and isinstance(call.args[0], ast.Name):
                bound = [a.id if isinstance(a, ast.Name) else None for a in call.args[1:]]
                bindings[node.targets[0].id] = (call.args[0].id, bound)
        return bindings

    def resolve_alias(self, name: str) -> str:
        """Follow ``h = g = ...`` chains (bounded by the alias count)."""
        aliases = self.name_aliases
        for _ in range(len(aliases) + 1):
            nxt = aliases.get(name)
            if nxt is None:
                return name
            name = nxt
        return name

    def factory_def(self, var: str, *, follow_aliases: bool = False) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The ``def`` of the factory that produced *var* (``c = make_counter()``)."""
        if follow_aliases:
            var = self.resolve_alias(var)
        source = self.functions.get(self.var_factories.get(var))
        if not source:
            return None
        try:
            parsed = ast.parse(source)
        except (SyntaxError, ValueError):
            return None
        if parsed.body and isinstance(parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parsed.body[0]
        return None

    def instance_class(self, var: str) -> str | None:
        """The notebook class *var* was constructed from, if it is one."""
        cls = self.var_factories.get(var)
        return cls if cls in self.classes else None

    def decorated_class(self, var: str) -> str | None:
        """The notebook class whose instance a class decorator bound *var* to."""
        cls = self.decorated_instances.get(var)
        return cls if cls in self.classes else None


# ---------------------------------------------------------------------------
# Cell effects: what the upstream checker resets on an isolated re-run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellEffects:
    """What running a cell writes, split by how an isolated re-run resets it.

    Every set leaves out names written by a ``# @cash: no-cache`` statement
    (:attr:`nocache`): those are meant to accumulate on a re-run.
    """

    #: Names the cell binds.
    outputs: frozenset[str] = frozenset()
    #: Names purely rebound with a fresh value (``name = ...``).
    reassigned: frozenset[str] = frozenset()
    #: Names changed in place, directly or through a called function, alias or view.
    mutated: frozenset[str] = frozenset()
    #: Receivers of a bare mutating method call, and arguments a called function
    #: mutates: restored to their cell-entry base.
    method_receivers: frozenset[str] = frozenset()
    #: Self-referential subscript/attribute writes (``df['a'] = df['a'] * 2``).
    selfref: frozenset[str] = frozenset()
    #: Swap / rotate targets (``a, b = b, a``).
    crossref_reassigned: frozenset[str] = frozenset()
    #: Functions, partials, closures and classes carrying state on their own
    #: object, whose producer must re-run.
    stateful_funcs: frozenset[str] = frozenset()
    #: Names written by a ``# @cash: no-cache`` statement.
    nocache: frozenset[str] = frozenset()
    #: Globals the cell changes without naming them (inside a called function,
    #: a partial, a ``reduce`` callback or an object-protocol method): they join
    #: the cell's inputs so their producer's base is restored.
    hidden_inputs: frozenset[str] = frozenset()
    #: Classes whose own state the cell changes (``Widget.count += 1`` in
    #: ``__init__``) that another cell changes too: re-running the class def
    #: would lose that cell's change, so it is not re-run, and an isolated
    #: re-run of this cell starts from the state it left last time.
    class_state_kept: frozenset[str] = frozenset()


def nocache_written_vars(cell_code: str, tree: ast.Module | None = None) -> frozenset[str]:
    """Variables written by a ``# @cash: no-cache`` statement in *cell_code*.

    ``no-cache`` means "always run fresh", so a self-modifying variable under
    it (``counter = counter + 1``) is meant to accumulate on a re-run, not be
    restored to its input. *tree* is ``ast.parse(cell_code)`` when the caller
    has it.
    """
    try:
        annotations = extract_annotations_for_statements(cell_code)
        if not annotations:
            return frozenset()
        if tree is None:
            tree = ast.parse(cell_code)
    except (SyntaxError, ValueError):
        return frozenset()

    written: set[str] = set()
    for node in tree.body:
        ann = annotations.get(getattr(node, "lineno", -1))
        if ann is None or not ann.no_cache:
            continue
        try:
            stmt_code = ast.unparse(node)
            _, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
            written |= outputs
            written |= set(analyze_statement(stmt_code, None).all_mutated_vars)
        except (SyntaxError, ValueError):
            continue
    return frozenset(written)


#: The statements whose branches are walked for :func:`control_structure_mutations`.
_COMPOUND = (ast.For, ast.While, ast.If, ast.With, ast.Try)


def _branches(node: ast.AST) -> list[ast.stmt]:
    """Every statement directly inside *node*: body, ``else``, handlers, ``finally``."""
    stmts = [*getattr(node, "body", []), *getattr(node, "orelse", [])]
    for handler in getattr(node, "handlers", []):
        stmts.extend(handler.body)
    stmts.extend(getattr(node, "finalbody", []))
    return stmts


def _loop_targets(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.For):
        return set()
    return {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}


@dataclass(frozen=True)
class _LoopNames:
    """How the engine reading a control structure resolves the names in it.

    The runtime and the simulation build one each, from the same kinds of
    answer: which names are modules, the namespace the kernel holds, and the
    source of a function called by name (the live one at run time, the
    notebook's cell text in the simulation). The rules that read it are the
    same for both, so both count the same names as changed.
    """

    is_module: Callable[[str], bool]
    namespace: Mapping[str, Any] | None = None
    resolve_source: SourceResolver | None = None

    def user_source(self, name: str) -> str | None:
        """The source of the function of the user's that *name* calls, or None."""
        if self.resolve_source is None:
            return None
        try:
            return self.resolve_source(name)
        except Exception:  # noqa: BLE001 - an unreadable callee is not one of the user's that is read
            return None

    def holds_a_path(self, receiver: ast.expr) -> bool:
        """Whether *receiver* is a module (``os``, ``glob``) or holds a
        ``pathlib`` path or an ``os.DirEntry``: what a listing method on it
        yields is read from the file system, fresh. A call (``Path(d)``)
        builds a fresh object no variable holds, so it counts too. Anything
        else, a tree of the user's with a ``walk()`` method, is not one."""
        if isinstance(receiver, ast.Call):
            return True
        dotted = _dotted(receiver)
        if not dotted:
            return False
        base, _, rest = dotted.partition(".")
        if not rest and self.is_module(base):
            return True
        if self.namespace is None or base not in self.namespace:
            return False
        value = self.namespace[base]
        for attr in rest.split(".") if rest else ():
            try:
                value = inspect.getattr_static(value, attr)
            except AttributeError:
                return False
        if isinstance(value, types.ModuleType):
            return True
        return _is_path_value(value)


def _is_path_value(value: Any) -> bool:
    """A ``pathlib`` path or an ``os.DirEntry``."""
    return isinstance(value, (pathlib.PurePath, os.DirEntry))


def control_structure_mutations(
    node: ast.AST,
    is_builtin: Callable[[str], bool],
    is_module: Callable[[str], bool],
    *,
    namespace: Mapping[str, Any] | None = None,
    resolve_source: SourceResolver | None = None,
) -> set[str]:
    """Names a control structure changes in place or accumulates into.

    Every branch counts, nested ones too (a loop's ``else`` included): each
    leaf statement's in-place mutations (``.append()``, ``d[k] = v``,
    ``obj.attr = v``) and self-referential reassignments (``total += b``,
    ``total = total + b``), which leave no in-place trace but carry the
    loop's result the same way. A bare method call whose method is not
    known to leave its receiver alone (``buf.write(s)``, ``acc.add(x)``)
    counts as a change to the receiver: the loop's per-statement path never
    classifies its body, so a receiver missed here keeps its pre-loop
    lineage and a statement reading it afterwards is served the value an
    earlier version of the loop left. So does a name handed to a function
    of the user's that changes that parameter (``add_tax(r)``,
    ``push(st, v)``), read through *resolve_source*. A module is never
    changed (*is_module*): ``os.remove(f)`` is not ``list.remove``. A loop
    target is a rebinding, not a mutation, so a loop's targets are left
    out within its body, and so are names *is_builtin* says are builtins.
    The condition of an ``if`` or ``while`` counts too
    (``condition_mutations``): it runs whichever branch is taken.

    What a library function does to the names it is handed
    (``heapq.heappush(h, x)``) the text cannot say: the runtime watches
    those (:func:`loop_watch_candidates`) and records what it saw.

    The runtime (``update_lineage_after_execution``) and the simulation
    (``VirtualLineage``) both call this, each with its own builtin rule,
    namespace and source of functions over the same lineage, so a loop
    bumps the same lineages on both sides.
    """
    names = _LoopNames(is_module, namespace, resolve_source)
    in_place, rebound, called = _branch_mutations(_branches(node), names)
    in_place |= _header_mutations(node, names)
    if isinstance(node, ast.For):
        in_place, rebound, called = _through_loop_targets(node, in_place, rebound, called, names)
    # ``os.remove(f)`` reads as ``list.remove`` on ``os``; a module's lineage
    # is its code, which no call through it changes.
    return {v for v in in_place | rebound if not is_builtin(v) and not is_module(v)}


def condition_mutations(test: ast.expr, resolve_source: SourceResolver | None = None) -> frozenset[str]:
    """Names the condition *test* of an ``if`` or ``while`` changes in place.

    A condition is code that runs: ``if opts.pop('debug', False):`` removes
    a key and ``if stack.pop() > 7:`` shortens the list, whichever branch is
    taken. Its result is used, so it counts as the statement ``_ = test``
    would: a method known to change its receiver (``pop``, ``add``,
    ``setdefault``), not every method whose result is dropped -- ``if
    name.startswith('a'):`` changes nothing -- and a name handed to a
    function of the user's that changes it (``if add(seen, x):``), read
    through *resolve_source*.
    """
    try:
        found = set(analyze_statement(ast.unparse(test), None).all_mutated_vars)
    except (SyntaxError, ValueError, AttributeError, TypeError):
        found = set()
    if resolve_source is not None:
        found |= _user_callee_changes(_as_statement(test), resolve_source)
    return frozenset(found)


def _as_statement(test: ast.expr) -> ast.Module:
    """``_ = test``: how a condition runs, as a statement."""
    assign = ast.Assign(targets=[ast.Name(id="_", ctx=ast.Store())], value=test, lineno=1)
    return ast.Module(body=[assign], type_ignores=[])


def _user_callee_changes(tree: ast.Module, resolve_source: SourceResolver) -> set[str]:
    """The names *tree* hands to a function of the user's that changes the
    matching parameter (``function_arg_mutations``). An unreadable callee is
    left to the runtime's watch (:func:`loop_watch_candidates`)."""
    try:
        return set(function_arg_mutations(tree, resolve_source))
    except Exception:  # noqa: BLE001 - what a callee does is then left to the watch
        return set()


def _header_mutations(node: ast.AST, names: _LoopNames | None = None) -> set[str]:
    """What the condition of *node* (an ``if`` or ``while``) changes."""
    if not isinstance(node, (ast.If, ast.While)):
        return set()
    return set(condition_mutations(node.test, names.resolve_source if names is not None else None))


_Changes = tuple[set[str], set[str], set[str]]


def _through_loop_targets(
    node: ast.For, in_place: set[str], rebound: set[str], called: set[str], names: _LoopNames | None = None
) -> _Changes:
    """A loop body's changes (`_branch_mutations`) as seen from outside the
    loop.

    A loop target is a rebinding, not a mutation, so it is left out. But a
    target changed in place (``for r in records: r['tax'] = ...``,
    ``for a in arrays: a += 1``, ``for r in records: add_tax(r)``) is an
    element of what the loop iterates over, so that changed: the names the
    iterable is built from (``records``, ``enumerate(rows)``, ``zip(a, b)``,
    ``d.values()``, ``axes.flat``, ``recs[1:]``, and the globals a function
    of the user's walks, ``active()``) count as changed in place instead. A
    bare method call on a target counts too, even one known to leave a frame
    alone (``for ax in axes: ax.plot(...)`` draws on each Axes): a call
    whose result the loop drops is made for what it does. A target only
    rebound (``line = line.strip()``) changes nothing the iterable holds.
    """
    targets = _loop_targets(node)
    if (in_place | called) & targets:
        in_place = in_place | _iterable_roots(node.iter, names)
    return in_place - targets, rebound - targets, called - targets


#: Calls whose items are made fresh for the loop -- paths or names read from
#: the file system -- so no variable holds them: ``old.unlink()`` in
#: ``for old in OUT.glob('*.txt')`` changes a file, not ``OUT``. A method
#: counts only on a module, a path or a fresh object (`_LoopNames.holds_a_path`):
#: ``tree.walk()`` on a tree of the user's yields what the tree holds.
_FILE_LISTING_METHODS = frozenset({"glob", "rglob", "iterdir", "walk"})
_FILE_LISTING_FUNCTIONS = frozenset(
    {"os.listdir", "os.scandir", "os.walk", "glob.glob", "glob.iglob", "listdir", "scandir", "glob", "iglob"}
)


def _dotted(node: ast.expr) -> str:
    """``os.listdir`` for an ``os.listdir`` reference, ``""`` for anything
    but a chain of names."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _lists_files(node: ast.expr, names: _LoopNames | None) -> bool:
    """Whether the call *node* lists the file system (`_FILE_LISTING_METHODS`,
    `_FILE_LISTING_FUNCTIONS`)."""
    if not isinstance(node, ast.Call):
        return False
    if _dotted(node.func) in _FILE_LISTING_FUNCTIONS:
        return True
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in _FILE_LISTING_METHODS):
        return False
    if names is None:
        return isinstance(func.value, ast.Call)
    return names.holds_a_path(func.value)


def _iterable_roots(node: ast.expr, names: _LoopNames | None = None) -> set[str]:
    """The names *node* reads that are not called: the containers a loop
    takes its elements from, not the functions that walk them. None when
    the items are fresh objects read from the file system (`_lists_files`).

    A function of the user's called by name (``for r in active():``) hands
    out what it reads: the globals its body reads join the roots, read
    through *names*' source of functions, as its arguments already do."""
    while isinstance(node, ast.Call) and _dotted(node.func) in {"sorted", "list", "tuple", "reversed"} and node.args:
        node = node.args[0]
    if _lists_files(node, names):
        return set()
    functions = {id(sub.func) for sub in ast.walk(node) if isinstance(sub, ast.Call)}
    roots = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and id(sub) not in functions}
    if names is not None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                source = names.user_source(sub.func.id)
                if source and resolve_function_def(sub.func.id, lambda _name, s=source: s) is not None:
                    roots |= _function_global_reads(source)
                elif (
                    names.namespace is not None
                    and isinstance(names.namespace.get(sub.func.id), (types.FunctionType, type(None)))
                    and users_callee(sub.func, names.namespace)
                ):
                    # A function of the user's whose body cannot be read (a
                    # lambda, one made by ``exec``) may hand out the items
                    # of any global it reaches.
                    roots |= _mutable_globals(names)
    return roots


#: Names IPython binds in the user namespace, not the user's data.
_IPYTHON_NAMES = frozenset({"In", "Out", "exit", "quit", "get_ipython"})


def _mutable_globals(names: _LoopNames) -> set[str]:
    """Every variable of the notebook a call could change in place."""
    namespace = names.namespace or {}
    return {
        name
        for name in namespace
        if not name.startswith("_") and name not in _IPYTHON_NAMES and _watchable(name, namespace, names.is_module)
    }


@functools.lru_cache(maxsize=1024)
def _function_global_reads(source: str) -> frozenset[str]:
    """The names a function's body reads and does not bind or call: the
    globals whose objects it can hand out (``records`` in ``def active():
    return [r for r in records if r['on']]``)."""
    fdef = resolve_function_def("f", lambda _name: source)
    if fdef is None:
        return frozenset()
    bound = set(all_param_names(fdef))
    callees = {id(sub.func) for sub in ast.walk(fdef) if isinstance(sub, ast.Call)}
    reads: set[str] = set()
    for sub in ast.walk(fdef):
        if isinstance(sub, ast.Name):
            if not isinstance(sub.ctx, ast.Load):
                bound.add(sub.id)
            elif id(sub) not in callees:
                reads.add(sub.id)
        elif isinstance(sub, ast.arg):
            bound.add(sub.arg)
    return frozenset(reads - bound - BUILTIN_NAMES)


def _branch_mutations(stmts: list[ast.stmt], names: _LoopNames) -> _Changes:
    """``(in place, self-referential rebinds, bare-call receivers)``: the
    names *stmts* change in place, those they rebind from themselves
    (``total += b``, ``total = total + b``), and the receivers of the method
    calls whose result they drop, known pure or not; a loop among them
    seen from outside it (`_through_loop_targets`)."""
    in_place: set[str] = set()
    rebound: set[str] = set()
    called: set[str] = set()
    is_module = names.is_module
    for stmt in stmts:
        if isinstance(stmt, _COMPOUND):
            changes = _branch_mutations(_branches(stmt), names)
            changes = (changes[0] | _header_mutations(stmt, names), changes[1], changes[2])
            if isinstance(stmt, ast.For):
                changes = _through_loop_targets(stmt, *changes, names)
            in_place |= changes[0]
            rebound |= changes[1]
            called |= changes[2]
            continue
        try:
            in_place.update(analyze_statement(ast.unparse(stmt), None).all_mutated_vars)
        except (SyntaxError, ValueError, AttributeError, TypeError):
            pass  # nothing the analysis can see; the rules below still apply
        rebound.update(selfref_reassignment_targets(stmt))
        in_place.update(_bare_call_receivers(stmt, is_module))
        called.update(_bare_call_receivers(stmt, is_module, pure_too=True))
        if names.resolve_source is not None:
            in_place.update(_user_callee_changes(ast.Module(body=[stmt], type_ignores=[]), names.user_source))
        # `%time acc.append(x)` in the body changes `acc` as the plain line does.
        inner_stmts = magic_python(ast.Module(body=[stmt], type_ignores=[])).body[1:]
        if inner_stmts:
            changes = _branch_mutations(inner_stmts, names)
            in_place |= changes[0]
            rebound |= changes[1]
            called |= changes[2]
    return in_place, rebound, called


def _bare_call_receivers(stmt: ast.stmt, is_module: Callable[[str], bool], *, pure_too: bool = False) -> set[str]:
    """Receivers of *stmt*'s bare method calls that may change them.

    A call whose result is dropped is made for what it does, so its receiver
    counts as changed unless the method is known pure (*pure_too* counts
    those as well) or only writes the receiver out to a file. A module is
    skipped: ``os.makedirs(p)`` changes no notebook variable.
    """
    calls = top_level_calls(ast.Module(body=[stmt], type_ignores=[]))
    receivers: set[str] = set()
    for base, method in calls.method_calls:
        if is_module(base) or method in RECEIVER_READONLY_WRITE_METHODS:
            continue
        if not pure_too and chain_is_pure(method, calls.inner_methods.get((base, method), frozenset())):
            continue
        receivers.add(base)
    return receivers


# ---------------------------------------------------------------------------
# What a loop hands to code whose effect the text cannot decide
# ---------------------------------------------------------------------------


#: The structures watched as a unit for what they hand to code whose effect
#: is unknown: a loop, and a ``with`` block (run as one unit). An ``if`` or a
#: ``try`` runs statement by statement, each watched on its own.
WATCHED_STRUCTURES = (ast.For, ast.While, ast.With)


def loop_watch_candidates(
    node: ast.AST,
    namespace: Mapping[str, Any],
    is_module: Callable[[str], bool],
    resolve_source: SourceResolver | None = None,
) -> frozenset[str]:
    """The names a loop (or a ``with`` block, `WATCHED_STRUCTURES`) hands to code whose effect on them its text cannot
    decide, to watch around the loop: what a library function is handed
    (``heapq.heappush(h, x)``, ``np.copyto(a, b)``, ``rng.shuffle(xs)``,
    ``if heapq.heappop(h) > t:``), what a method is handed, the receiver of
    a method whose result is used (``x = dq.popleft()``), and what a
    function of the user's whose source cannot be read is handed. A loop
    target counts as the names its loop iterates (``for xs in lists:
    random.shuffle(xs)`` hands over items of ``lists``).

    A function of the user's that is read counts for what its body hands
    to such code: the arguments for those parameters, and the globals
    (``def push(h, x): heapq.heappush(h, x)``, `function_handoffs`).

    Not candidates: what :func:`control_structure_mutations` already decides
    from the text (a bare method call's receiver), what a call known only to read is handed (``print(df)``,
    ``len(xs)``, ``sum(xs)``), modules, classes, functions and immutable
    values. The runtime fingerprints the candidates before and after the
    loop runs and records which changed, per loop source; the simulation and
    a restored loop apply that record.
    """
    if not isinstance(node, WATCHED_STRUCTURES):
        return frozenset()
    names = _LoopNames(is_module, namespace, resolve_source)
    found: set[str] = set()
    targets: dict[str, set[str]] = {}
    _watch_statements([node], names, found, targets)
    # What a ``with`` binds (``as f``) is the object its context manager
    # hands back, made as the block starts: a name left from an earlier run
    # holds another object.
    found -= _with_bound_names(node)
    resolved: set[str] = set()
    pending = list(found)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in targets:
            pending.extend(targets[name])
        else:
            resolved.add(name)
    return frozenset(name for name in resolved if _watchable(name, namespace, is_module))


def _with_bound_names(node: ast.AST) -> set[str]:
    """The names the ``with`` statements in *node* bind with ``as``."""
    return {
        target.id
        for stmt in ast.walk(node)
        if isinstance(stmt, (ast.With, ast.AsyncWith))
        for item in stmt.items
        if item.optional_vars is not None
        for target in ast.walk(item.optional_vars)
        if isinstance(target, ast.Name)
    }


def loop_target_roots(
    node: ast.For,
    namespace: Mapping[str, Any] | None,
    is_module: Callable[[str], bool],
    resolve_source: SourceResolver | None = None,
) -> dict[str, frozenset[str]]:
    """Each target of the loop *node*, with the names its values come out of
    (``records`` for ``r`` in ``for r in records:``): a change to the target
    in place is a change to what those hold (`_through_loop_targets`)."""
    roots = frozenset(_iterable_roots(node.iter, _LoopNames(is_module, namespace, resolve_source)))
    return {target: roots for target in _loop_targets(node)}


def statement_watch_candidates(
    tree: ast.Module | None,
    namespace: Mapping[str, Any],
    is_module: Callable[[str], bool],
    resolve_source: SourceResolver | None = None,
) -> frozenset[str]:
    """What the one statement *tree* of a loop or branch body hands to code
    whose effect on it its text cannot decide (:func:`loop_watch_candidates`
    for one statement): ``x = heapq.heappop(h)``, ``x = dq.popleft()``."""
    if tree is None or len(tree.body) != 1 or isinstance(tree.body[0], _COMPOUND):
        return frozenset()
    names = _LoopNames(is_module, namespace, resolve_source)
    found = _handed_to_unknown_code(tree, names, bare=True)
    return frozenset(name for name in found if _watchable(name, namespace, is_module))


def comprehension_watch_candidates(
    tree: ast.Module | None,
    namespace: Mapping[str, Any],
    is_module: Callable[[str], bool],
    resolve_source: SourceResolver | None = None,
) -> frozenset[str]:
    """What the comprehensions and lambdas of the statement *tree* hand to
    code whose effect on it its text cannot decide, item by item:
    ``[heapq.heappush(h, x) for x in xs]``, ``any(rng.shuffle(r) for r in
    rows)``, ``map(lambda x: heapq.heappush(h, x), xs)``. A comprehension
    is a loop written as an expression, so what its element hands over is
    watched as a loop body's statement is (:func:`loop_watch_candidates`); a
    comprehension target counts as what it iterates. Not the statement's
    other calls: a library call whose result is kept returns its answer
    (``call_arguments`` decides those)."""
    if tree is None:
        return frozenset()
    names = _LoopNames(is_module, namespace, resolve_source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            parts = [node.elt]
        elif isinstance(node, ast.DictComp):
            parts = [node.key, node.value]
        elif isinstance(node, ast.Lambda):
            parts = [node.body]
        else:
            continue
        for gen in getattr(node, "generators", ()):
            parts.extend(gen.ifs)
        for part in parts:
            found |= _handed_to_unknown_code(_as_statement(part), names, bare=False)
    if not found:
        return frozenset()
    iterated = iterated_sources(tree)
    roots: set[str] = set()
    for name in found:
        roots |= iterated.get(name, frozenset({name}))
    # A random generator's draws are followed as a carrier's are.
    return frozenset(
        name
        for name in roots
        if _watchable(name, namespace, is_module) and rng_carrier_kind(namespace.get(name)) is None
    )


def _watch_statements(stmts: list[ast.stmt], names: _LoopNames, found: set[str], targets: dict[str, set[str]]) -> None:
    """Gather into *found* what each of *stmts* hands to unknown code, and
    into *targets* the names each loop among them iterates, per target. A
    ``def`` or ``class`` in the loop runs nothing until called."""
    for stmt in stmts:
        if isinstance(stmt, _COMPOUND):
            if isinstance(stmt, ast.For):
                roots = _iterable_roots(stmt.iter, names)
                for target in _loop_targets(stmt):
                    targets.setdefault(target, set()).update(roots)
            if isinstance(stmt, (ast.If, ast.While)):
                found |= _handed_to_unknown_code(_as_statement(stmt.test), names, bare=False)
            if isinstance(stmt, ast.With):
                for item in stmt.items:
                    found |= _handed_to_unknown_code(_as_statement(item.context_expr), names, bare=False)
            _watch_statements(_branches(stmt), names, found, targets)
            continue
        found |= _handed_to_unknown_code(ast.Module(body=[stmt], type_ignores=[]), names, bare=True)


#: Builtins that read what they are handed and change none of it.
_READING_BUILTINS = ARGUMENT_READING_BUILTINS


def _handed_to_unknown_code(tree: ast.Module, names: _LoopNames, *, bare: bool) -> set[str]:
    """The names the one statement *tree* hands to code whose effect on
    them its text cannot decide (see :func:`loop_watch_candidates`)."""
    namespace = names.namespace or {}
    stmt = tree.body[0] if tree.body else None
    outer = stmt.value if bare and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) else None
    calls = calls_with_lambdas(tree, control_bodies=False)
    # A call a chain goes on from (``df.groupby('a')`` in
    # ``df.groupby('a').sum()``) is judged with the chain's last method.
    chained = {id(inner) for call in calls for inner in _chain_calls(call.func)}
    out: set[str] = set()
    for call in calls:
        func = call.func
        if isinstance(func, ast.Name) and names.user_source(func.id):
            # Its source says what it changes itself; what it hands to code
            # whose effect is unknown (``def push(h, x): heapq.heappush(h,
            # x)``) is watched as if handed here.
            out |= _user_function_handoffs(call, func.id, names)
            continue
        if isinstance(func, ast.Attribute) and call is not outer and id(call) not in chained:
            base = _receiver_root(func.value)
            inner = frozenset(c.func.attr for c in _chain_calls(func) if isinstance(c.func, ast.Attribute))
            if base and func.attr not in RECEIVER_READONLY_WRITE_METHODS and not chain_is_pure(func.attr, inner):
                out.add(base)
        if isinstance(func, ast.Name) and func.id in _READING_BUILTINS and func.id not in namespace:
            continue
        if isinstance(func, ast.Attribute) and _stores_what_it_is_handed(func, namespace):
            continue
        args = [*call.args, *(kw.value for kw in call.keywords)]
        arg_names = [arg.id for arg in args if isinstance(arg, ast.Name)]
        # Read before the loop runs: a name the loop binds (``ax`` in ``for
        # ax in axes: ax.plot(df)``) is not yet in the namespace.
        if call_only_reads(call, arg_names, namespace, bound_by_now=False):
            continue
        for arg in args:
            out |= argument_roots(arg, {})
    return out


def _user_function_handoffs(call: ast.Call, name: str, names: _LoopNames) -> set[str]:
    """What *call*, to the function of the user's *name*, hands to code
    whose effect on it is unknown: its arguments for the parameters the
    function hands on, and the globals it hands on (`function_handoffs`)."""
    fdef = resolve_function_def(name, names.user_source)
    if fdef is None:
        return set()
    params, free = function_handoffs(fdef, names.namespace or {}, names.user_source, frozenset({name}))
    return handed_to_params(call, fdef, params, {}) | set(free)


def _stores_what_it_is_handed(func: ast.Attribute, namespace: Mapping[str, Any]) -> bool:
    """A method known to change its receiver (``acc.append(x)``), or any
    method of a builtin container or scalar: it keeps or reads what it is
    handed, and changes none of it."""
    if func.attr in MUTATING_METHODS:
        return True
    base = _receiver_root(func.value)
    return base is not None and isinstance(namespace.get(base), _KNOWN_METHOD_TYPES)


def _chain_calls(func: ast.expr) -> list[ast.Call]:
    """The calls a method chain makes on the way to *func*:
    ``df.groupby('a')`` for ``df.groupby('a').sum``."""
    found: list[ast.Call] = []
    node: ast.AST = func
    while True:
        if isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            found.append(node)
            node = node.func
        else:
            return found


def _watchable(name: str, namespace: Mapping[str, Any], is_module: Callable[[str], bool]) -> bool:
    """Whether a call could change *name*'s value in place: bound, and not
    a module, class, function or immutable value."""
    if is_module(name) or name not in namespace:
        return False
    value = namespace[name]
    if isinstance(value, (types.ModuleType, type, *_IMMUTABLE_VALUES)):
        return False
    return not inspect.isroutine(value)


#: Values no call can change in place.
_IMMUTABLE_VALUES = (int, float, complex, str, bytes, bool, type(None), frozenset, tuple, range)


def _is_live_ndarray(val: Any) -> bool:
    """A numpy ndarray (so ``v = val[slice]`` is a view, not a copy), checked
    without importing numpy."""
    t = type(val)
    return t.__name__ == "ndarray" and t.__module__.split(".")[0] == "numpy"


def cell_effects(cell_code: str, sources: NotebookSources, namespace: Mapping[str, Any]) -> CellEffects:
    """Everything running *cell_code* writes, for the upstream checker.

    *sources* resolves the functions, classes and bindings the cell refers to;
    it is only read when the cell calls, subscripts, deletes or enters
    something that could reach one. *namespace* is read only to tell a numpy
    view from a copy. A cell that does not parse (a magic) writes nothing
    this can see.
    """
    try:
        facts = _CellFacts(cell_code, ast.parse(cell_code))
        writes = _CellWrites.of(facts, sources, namespace)
        # Only what the cell hands over is read from its objects, under the
        # budget (`_CellWrites.of`). The rest is read from the cell and the
        # functions it names and is answered in full, as a statement's is
        # (`_effects_past_the_budget`): a budget that ran out above stays
        # tripped, and its raise here ran the whole cell without cash.
        with no_analysis_budget():
            writes.add_argument_mutations(facts, sources)
            writes.add_callee_state(facts, sources)
            writes.add_object_protocol(facts, sources)
            writes.add_aliases_and_views(facts, namespace)
        return writes.effects(facts)
    except (SyntaxError, ValueError):
        return CellEffects()


class _CellFacts:
    """One cell, parsed once, and what the analyses read from it, each worked
    out once on first use."""

    def __init__(self, code: str, tree: ast.Module) -> None:
        self.code = code
        self.tree = tree

    @functools.cached_property
    def _flow_tree(self) -> ast.Module | None:
        """The tree the flow analysis would parse itself: the same one, unless
        stripping magics changes the text."""
        return self.tree if CodeAnalyzer.strip_magics(self.code) == self.code else None

    @functools.cached_property
    def outputs(self) -> set[str]:
        return CodeAnalyzer.analyze_code_block(self.code, tree=self._flow_tree)[1]

    @functools.cached_property
    def reassigned(self) -> set[str]:
        return CodeAnalyzer.reassigned_names(self.code, self._flow_tree)

    @functools.cached_property
    def calls_something(self) -> bool:
        return bool(called_names(self.tree))

    @functools.cached_property
    def own_mutated(self) -> frozenset[str]:
        """Names the cell's own text changes in place."""
        return analyze_statement(self.code, self.tree).all_mutated_vars

    @functools.cached_property
    def nocache(self) -> frozenset[str]:
        return nocache_written_vars(self.code, self.tree)


@dataclass
class _CellWrites:
    """The sets `CellEffects` is built from, filled one channel at a time.
    Every step leaves out what a ``# @cash: no-cache`` statement writes."""

    mutated: set[str]
    method_receivers: set[str]
    selfref: set[str]
    #: The globals a called function mutates (empty when the cell calls nothing).
    callee_globals: frozenset[str]
    stateful: set[str] = field(default_factory=set)
    hidden: set[str] = field(default_factory=set)
    class_state_kept: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, facts: _CellFacts, sources: NotebookSources, namespace: Mapping[str, Any]) -> _CellWrites:
        # Globals a callee mutates count as mutated here, exactly like an
        # inline mutation, so the reset covers them. They are deliberately NOT
        # outputs: the checker reads `outputs` as "written by the cell, not an
        # input to restore", which would disable the very reset that makes the
        # statement's key converge. The callables the cell hands to a call
        # (``s.apply(ops['dbl'])``, ``s.apply(tracker.record)``) count as
        # called, and the objects they change as mutated.
        try:
            handed = handed_callables(facts.tree, namespace)
            extra_sources = handed.sources
            receivers = set(handed.receivers)
        except AnalysisBudgetExceeded:
            # What the cell hands over could not be read within its analysis
            # budget (`analysis_budget`): any function or method of the
            # notebook may be among it, and change what it is handed or
            # what it calls a method on.
            extra_sources = (*sources.functions.values(), *sources.classes.values())
            receivers = handed_or_called_on(facts.tree)
        with no_analysis_budget():
            callee_globals = (
                callee_global_mutations(facts.tree, sources.functions.get, extra_sources=extra_sources)
                if facts.calls_something or extra_sources
                else frozenset()
            )
        if receivers:
            callee_globals = callee_globals | capturable_globals(frozenset(receivers), namespace)
        nocache = facts.nocache
        return cls(
            mutated=set(facts.own_mutated | callee_globals) - nocache,
            # Receivers of a bare method call (``b.items.append``): such
            # no-output statements skip the per-statement cache, so a
            # lineage-carrying receiver would accumulate on an isolated
            # re-run. Method receivers only, so ``df['col'] = ...`` keeps its
            # per-statement cache.
            method_receivers=set(standalone_method_mutation_receivers(facts.tree)) - nocache,
            # Self-referential subscript/attr writes (``df['a'] = df['a'] * 2``,
            # ``df.iloc[i, j] += x``) are not idempotent. A new column read
            # from other columns is not self-referential and keeps its cache.
            selfref=set(selfref_inplace_write_vars(facts.tree)) - nocache,
            callee_globals=callee_globals,
        )

    def add_argument_mutations(self, facts: _CellFacts, sources: NotebookSources) -> None:
        """A variable passed to a helper that mutates that parameter
        (``def add(d): d.append(x)`` + ``add(data)`` or ``n = add(data)``) is
        reset like a receiver."""
        if not facts.calls_something:
            return  # nothing to look up: the notebook is not read
        arg_muts = function_arg_mutations(facts.tree, sources.functions.get) - facts.nocache
        self.mutated |= arg_muts
        self.method_receivers |= arg_muts

    def add_callee_state(self, facts: _CellFacts, sources: NotebookSources) -> None:
        """State a called function changes without the cell naming it."""
        if not facts.calls_something:
            return
        tree, nocache, functions = facts.tree, facts.nocache, sources.functions.get
        # A global a called function mutates (``def bump(): global g; g += 1``)
        # joins the inputs so its producer's base is restored. Every call
        # counts, not only a bare-``Expr`` one: the statement keys on the
        # global's pre-state, so without the reset the value it produced would
        # be the next run's key -- a cell that re-executes and accumulates
        # forever.
        global_muts = self.callee_globals - nocache
        self.mutated |= global_muts
        self.hidden |= global_muts
        # State on the function object itself (a mutated mutable default, a
        # function attribute, a memoizer): re-run its ``def``.
        self.stateful |= set(stateful_self_functions(tree, functions)) - nocache
        # A closure (``c = make_counter()``) whose factory's inner function
        # mutates factory-local state: re-run the factory call.
        self.stateful |= set(stateful_closure_vars(tree, sources.factory_def)) - nocache
        # Mutation through functools.partial or a functools.reduce callback.
        partials = sources.partial_bindings.get
        partial_muts = (
            partial_arg_mutations(tree, partials, functions) | reduce_free_mutations(tree, functions)
        ) - nocache
        self.mutated |= partial_muts
        self.hidden |= partial_muts
        # A partial that bound a mutated argument holds the argument's object:
        # re-bind it too.
        self.stateful |= set(mutating_partials(tree, partials, functions)) - nocache

    def add_object_protocol(self, facts: _CellFacts, sources: NotebookSources) -> None:
        op = _object_protocol_effects(facts.code, facts.tree, sources, facts.nocache)
        if op is None:
            return
        op_free, op_receivers, op_class_defs, op_init_free, op_kept = op
        self.mutated |= op_free | op_receivers | op_init_free
        self.hidden |= op_free | op_init_free
        self.method_receivers |= op_receivers
        self.stateful |= op_class_defs
        self.class_state_kept |= op_kept

    def add_aliases_and_views(self, facts: _CellFacts, namespace: Mapping[str, Any]) -> None:
        tree, nocache = facts.tree, facts.nocache
        # ``y = x`` shares x's object, so ``y.append`` also mutates x. The
        # selfref set is column-scoped, so an aliased NEW-column write keeps
        # its cache.
        self.mutated |= aliased_sources(tree, facts.own_mutated) - nocache
        self.selfref |= aliased_sources(tree, self.selfref) - nocache
        self.method_receivers |= aliased_sources(tree, self.method_receivers) - nocache
        # ``v = arr[slice]`` on a numpy array is a view: mutating v mutates
        # arr. A list slice is a copy, hence the live-value check.
        view_bindings = subscript_view_bindings(tree)
        if view_bindings:
            self.mutated |= {
                base
                for alias, base in view_bindings.items()
                if alias in facts.own_mutated and _is_live_ndarray(namespace.get(base))
            } - nocache

    def effects(self, facts: _CellFacts) -> CellEffects:
        nocache = facts.nocache
        return CellEffects(
            outputs=frozenset(facts.outputs),
            reassigned=frozenset(facts.reassigned - nocache),
            # A kept class's state is put back before a re-run of the cell
            # (`class_state.ClassState`), whichever channel changes it: one
            # passed to a call (``id(Widget)``, a handed callable that may
            # construct it) re-running its def instead would drop what the
            # other cells' instances did to it.
            mutated=frozenset(self.mutated - self.class_state_kept),
            method_receivers=frozenset(self.method_receivers),
            selfref=frozenset(self.selfref),
            # ``a, b = b, a`` reads its own pre-cell value but, re-run alone,
            # holds the swapped output: lineage-invisible, so reset it by rule.
            crossref_reassigned=frozenset(crossref_reassigned_vars(facts.tree) - nocache),
            stateful_funcs=frozenset(self.stateful),
            nocache=nocache,
            hidden_inputs=frozenset(self.hidden),
            class_state_kept=frozenset(self.class_state_kept),
        )


def _object_protocol_effects(
    cell_code: str, tree: ast.Module, sources: NotebookSources, nocache: frozenset[str]
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]] | None:
    """Hidden state reached through the object protocol, per reset channel:
    ``(free_vars, receivers, class_defs, init_subclass_free_vars, kept)``,
    *kept* the classes whose def is not re-run because another cell changes
    them too, or because their objects change them (`CellEffects.class_state_kept`).

    A ``with``, a custom-dunder operation, a constructor, a decorated call or a
    method whose body mutates hidden state. None when the cell has nothing
    that could dispatch to such a method.
    """
    # Any call / subscript / del / with can dispatch to a custom method; a
    # based ``class Sub(Base):`` runs the base's ``__init_subclass__``.
    triggered = any(
        isinstance(n, (ast.Call, ast.Subscript, ast.Delete, ast.With, ast.AsyncWith)) for n in ast.walk(tree)
    ) or any(isinstance(n, ast.ClassDef) and n.bases for n in tree.body)
    if not triggered:
        return None

    def op_var_factory(name):
        return sources.factory_def(name, follow_aliases=True)

    def resets_of(cell_tree: ast.Module) -> ObjectProtocolResets:
        # One call shape for this cell and every other cell, so the sole-
        # mutator check below resolves exactly what this cell resolves
        # (instances, ``@Counter def task`` decorations, factories).
        return object_protocol_mutations(
            cell_tree,
            sources.classes.get,
            sources.instance_class,
            sources.functions.get,
            op_var_factory,
            decorated_class=sources.decorated_class,
        )

    resets = resets_of(tree)
    free = resets.free_vars - nocache
    receivers = resets.receivers - nocache
    class_defs = resets.class_defs - nocache
    # An ``__init_subclass__`` registry hides its mutation behind class
    # creation, so the simulator's content-base guard cannot see a sibling
    # subclass cell also appending to it; it gets the cross-cell check below.
    init_free = resets.init_subclass_free_vars - nocache
    kept: frozenset[str] = frozenset()
    if receivers or class_defs or init_free:
        # Re-deriving a receiver or re-running a class def is safe only when
        # this cell is the SOLE in-place mutator: another cell's contribution
        # (``dag.add_edge`` in a setup cell, a second ``Widget()`` bumping
        # ``Widget.count``) would be lost, even on the first run.
        other_receivers: set[str] = set()
        other_class_defs: set[str] = set()
        other_init_free: set[str] = set()
        seen_current = False
        for code in sources.cells:
            if not seen_current and code == cell_code:
                seen_current = True
                continue
            try:
                other = resets_of(ast.parse(code))
            except (SyntaxError, ValueError):
                continue
            other_receivers |= other.receivers
            other_class_defs |= other.class_defs
            other_init_free |= other.init_subclass_free_vars
        receivers -= other_receivers
        # A class changed through its objects (``type(self).count += 1``)
        # is kept too: re-running its def would leave the instances made
        # before on the old class, still changing that one.
        kept = (class_defs & other_class_defs) | changed_through_their_objects(class_defs, sources.classes.get)
        class_defs -= kept
        init_free -= other_init_free
    return free, receivers, class_defs, init_free, kept


# ---------------------------------------------------------------------------
# Statement effects: what the runtime and the simulation key and bump
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatementEffects:
    """What one statement reads and writes, as both engines must see it.

    A callee's writes to globals show up twice, for two questions.
    ``outputs`` answers "what does this statement produce": every call counts,
    including one in a loop or branch body, so each written global gets a
    producer and a lineage that advances. ``callee_globals`` answers "which
    writes does this statement own, to replay or skip": a loop or branch is one
    unit to the simulation and the accumulator machinery, so its body's writes
    belong to it and not to one body statement. Both engines read both fields
    the same way, which keeps their keys equal.
    """

    #: Names the statement reads.
    inputs: frozenset[str]
    #: Names it binds, plus the globals any callee it calls mutates in place.
    outputs: frozenset[str]
    #: Notebook globals a callee mutates in place, counting only calls outside
    #: control-structure bodies. Empty for a statement in a control-structure
    #: body: the loop or branch owns its body's writes.
    callee_globals: frozenset[str]
    #: Variables handed to a bare call of a user function that mutates the
    #: matching parameter in place (``def add(d): d.append(x)`` + ``add(data)``).
    arg_mutations: frozenset[str]
    #: Why the callees could not be analysed, when they could not. The
    #: statement then runs uncached: what it writes is unknown.
    unanalysed: tuple[str, ...] = ()


def is_module_name(name: str, namespace: Mapping[str, Any], virtual_modules: Iterable[str] = ()) -> bool:
    """Whether *name* is bound to a module: live in *namespace*, or, when it is
    not bound there yet, among the *virtual_modules* a simulation bound."""
    if isinstance(namespace.get(name), types.ModuleType):
        return True
    return name not in namespace and name in virtual_modules


def _resolve_once(resolve_source: SourceResolver) -> SourceResolver:
    """*resolve_source*, answering each name from its first answer.

    A resolver that raises is asked again next time, so each caller still
    sees the exception and handles it its own way.
    """
    answers: dict[str, str | None] = {}

    def resolve(name: str) -> str | None:
        if name not in answers:
            answers[name] = resolve_source(name)
        return answers[name]

    return resolve


def statement_effects(
    code: str,
    tree: ast.Module | None,
    *,
    namespace: Mapping[str, Any],
    resolve_source: SourceResolver,
    control_body: bool = False,
    virtual_modules: frozenset[str] | set[str] = frozenset(),
) -> StatementEffects:
    """What *code* reads and writes, resolving called functions through
    *resolve_source*.

    *tree* is ``ast.parse(code)``, or None when that fails (a magic, a
    top-level ``await``): the inputs and outputs are then read from a
    magic-tolerant parse, and ``callee_globals`` and ``arg_mutations`` are
    empty. *virtual_modules* are names the simulation bound to modules that
    the kernel does not hold yet.

    Each called name is resolved once, however many of the questions above
    ask about it.
    """
    resolve_source = _resolve_once(resolve_source)
    callee_globals: frozenset[str] = frozenset()
    arg_mutations: frozenset[str] = frozenset()
    unreadable: tuple[str, ...] = ()
    try:
        with one_reading():
            inputs, outputs = CodeAnalyzer.analyze_code_block(
                code, tree=tree, resolve_source=resolve_source, user_ns=namespace
            )
            handed = handed_callables(tree, namespace, scope="no_control_bodies") if not control_body else None
        if handed is not None:
            callee_globals = (
                callee_global_mutations(tree, resolve_source, scope="no_control_bodies", extra_sources=handed.sources)
                | handed.receivers
            )
            if namespace is not None:
                callee_globals = capturable_globals(callee_globals, namespace)
            unreadable = handed.unreadable
        arg_mutations = frozenset(
            v for v in function_arg_mutations(tree, resolve_source) if not is_module_name(v, namespace, virtual_modules)
        )
    except AnalysisBudgetExceeded:
        return _effects_past_the_budget(code, tree, namespace, resolve_source, control_body, virtual_modules)
    except Exception as exc:  # noqa: BLE001 - unknown writes must not read as no writes
        # What the called functions write is unknown, so the statement keeps
        # its own reads and writes and is marked to run uncached.
        inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=tree, user_ns=namespace)
        reason = analysis_failed("analyse the functions this statement calls", exc)
        return StatementEffects(frozenset(inputs), frozenset(outputs), frozenset(), frozenset(), (reason,))
    # A user callable handed to a call whose source cannot be read: what it
    # changes is unknown, so the statement runs uncached.
    return StatementEffects(frozenset(inputs), frozenset(outputs), callee_globals, arg_mutations, unreadable)


def _effects_past_the_budget(
    code: str,
    tree: ast.Module | None,
    namespace: Mapping[str, Any],
    resolve_source: SourceResolver,
    control_body: bool,
    virtual_modules: frozenset[str] | set[str],
) -> StatementEffects:
    """`statement_effects` when reading what the statement's calls are
    handed ran past the statement's analysis budget (`analysis_budget`).

    Everything but that reading is still answered in full -- it is read
    from the statement and the functions it names, never from the objects
    -- so the globals a function it calls writes are still outputs. What the
    callables it hands over write is the part left unknown: the statement
    runs uncached (*unanalysed*), as one handed a callable whose source
    cannot be read, and its outputs are keyed by value (the processor).
    """
    with no_analysis_budget():
        inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=tree, user_ns=namespace)
        callee_globals: frozenset[str] = frozenset()
        arg_mutations: frozenset[str] = frozenset()
        if tree is not None:
            if not control_body:
                callee_globals = callee_global_mutations(tree, resolve_source, scope="no_control_bodies")
            every = callee_global_mutations(tree, resolve_source)
            if namespace is not None:
                callee_globals = capturable_globals(callee_globals, namespace)
                every = capturable_globals(every, namespace) | notebook_global_rebinds(tree, resolve_source, namespace)
            outputs = set(outputs) | set(every)
            arg_mutations = frozenset(
                v
                for v in function_arg_mutations(tree, resolve_source)
                if not is_module_name(v, namespace, virtual_modules)
            )
    return StatementEffects(
        frozenset(inputs), frozenset(outputs), callee_globals, arg_mutations, (ANALYSIS_BUDGET_REASON,)
    )


# ---------------------------------------------------------------------------
# Receivers: which objects a statement's calls change in place
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiverClasses:
    """A statement's call receivers and arguments, decided as far as the
    rules and the recorded verdict go.

    What is left over depends on what the engine can do: the runtime watches
    the statement run, the simulation cannot.
    """

    #: Changed in place: a known mutator, a draw on a Figure/Axes, a fit, a
    #: module setting, or a name the recorded verdict lists.
    mutated: frozenset[str] = frozenset()
    #: Method receivers no rule decides and no verdict covers. The runtime
    #: observes or assumes them; the simulation assumes they change.
    unknown_receivers: frozenset[str] = frozenset()
    #: Call arguments no verdict covers. The runtime fingerprints them;
    #: the simulation leaves them alone, since treating every ``print(df)``
    #: as a change would bump ``df`` for every reader.
    unknown_args: frozenset[str] = frozenset()
    #: Modules counted as changed by a ``set_*`` call that is also handed
    #: names, while no verdict says what the call changed, with those names:
    #: ``nx.set_node_attributes(G, region, "region")`` sets on ``G``. The
    #: runtime drops the module once it sees a handed name change
    #: (`MutationRouting.observe`); the recorded verdict then decides.
    provisional: Mapping[str, frozenset[str]] = field(default_factory=dict)


#: Receivers whose methods are known: the builtin containers and scalars,
#: whose in-place methods (``pop``, ``setdefault``) the mutation analysis
#: names, and the data-library values (frames, arrays), whose methods return
#: new objects.
_KNOWN_METHOD_TYPES = (list, dict, set, frozenset, tuple, str, bytes, bytearray, int, float, complex, bool, range)


def captured_call_receiver_names(tree: ast.Module | None) -> frozenset[str]:
    """The receivers of a method call whose result is bound, live or not:
    `captured_call_receivers` without the namespace filter."""
    return frozenset(
        base
        for base, method in assigned_method_call_receivers(tree)
        if method not in RECEIVER_READONLY_WRITE_METHODS and not chain_is_pure(method, frozenset())
    )


def captured_call_receivers(tree: ast.Module | None, namespace: Mapping[str, Any]) -> frozenset[str]:
    """Receivers of a method call whose result is bound, which the call could
    change in place: ``history = net.fit(X, epochs=3)``, ``out = trainer.train()``.

    A training or stepping call that returns something (a history, a loss,
    ``self``) changes its receiver as much as a bare ``net.fit(X)`` does, and
    a hit that restores only the result leaves the receiver untrained. These
    are watched like the arguments of a bare call: the runtime fingerprints
    them around the statement's first run and records which changed, and the
    simulation replays that verdict. One definition for both engines.

    Not candidates: modules, classes, functions and plain values; the builtin
    containers and data-library values (frames, arrays), whose methods are
    known; random generators, whose draws are followed as carriers; a known
    pure method (``m = df.mean()``) or one that writes a file; and a
    Figure/Axes or an estimator being fitted, which `classify_receivers`
    already counts as changed.
    """
    out: set[str] = set()
    for base, method in assigned_method_call_receivers(tree):
        if base not in namespace or method in RECEIVER_READONLY_WRITE_METHODS:
            continue
        value = namespace[base]
        if isinstance(value, (types.ModuleType, type, _KNOWN_METHOD_TYPES)) or inspect.isroutine(value):
            continue
        if builtin_hash_family(type(value)) is not None or rng_carrier_kind(value) is not None:
            continue
        if chain_is_pure(method, frozenset()) or is_pandas_plot_call(method, value):
            continue
        if receiver_is_identity_coupled(value) or fits_its_receiver(method, value):
            continue
        out.add(base)
    return frozenset(out)


def drawn_on_arguments(tree: ast.Module | None, namespace: Mapping[str, Any]) -> frozenset[str]:
    """Figures and Axes handed to a call, which draws on them: by a method
    (``df.plot(ax=ax)``) or by a plain function (``forest(axes[0], df)``)."""
    return _drawn_on(top_level_calls(tree), namespace)


def _drawn_on(calls: TopLevelCalls, namespace: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(name for name in calls.argument_bases if receiver_is_identity_coupled(namespace.get(name)))


def classify_receivers(
    tree: ast.Module | None,
    namespace: Mapping[str, Any],
    load_verdict: Callable[[], Iterable[str] | None],
    *,
    arguments: Iterable[str],
    virtual_modules: Iterable[str] = (),
) -> ReceiverClasses:
    """Decide which receivers and arguments of *tree*'s calls change in place.

    One rule set for the runtime and the simulation, so both bump the same
    lineages (the unified-key rule). *load_verdict* returns what the runtime
    recorded for this statement (None when it has not run); it is called only
    when the statement has a receiver or argument to decide, since the
    simulation reads it from the backend. *arguments* are the
    call arguments the engine considers (see ``call_arguments``).
    A module is never a receiver: ``pd.set_option(...)`` is a module
    function call, counted only as a change to a setting the module keeps.
    """
    calls = top_level_calls(tree)
    candidates = calls.method_calls
    # A captured return (``counts, bins, _ = ax.hist(...)``) is an assignment,
    # so it is not a bare-``Expr`` candidate; it must not hit the early return.
    assigned = calls.assigned_method_calls
    drawn_args = _drawn_on(calls, namespace)
    args = frozenset(arguments) - drawn_args
    tier1 = calls.mutation_receivers
    changed = calls.changed_receivers
    if not candidates and not assigned and not drawn_args and not args and not changed:
        return ReceiverClasses()
    inner = calls.inner_methods
    settings = calls.setting_receivers
    verdict = load_verdict()
    mutated: set[str] = set()
    unknown: set[str] = set()
    provisional: dict[str, frozenset[str]] = {}
    for base, method in candidates:
        receiver = namespace.get(base)
        if is_module_name(base, namespace, virtual_modules):
            if base not in settings:
                continue
            # pd.set_option, plt.rcParams.update. A ``set_*`` call handed
            # names (``nx.set_node_attributes(G, ...)``) that was seen
            # changing one of them set on it, not on the module: counted as
            # a change to the module, every re-run moved the lineage of
            # every statement reading ``nx``, and a restart that ran the
            # call fewer times missed them all.
            handed = frozenset(
                name
                for name in calls.setting_arguments.get(base, ())
                if not is_module_name(name, namespace, virtual_modules)
            )
            if handed and verdict is not None and handed & set(verdict) and base not in verdict:
                continue
            mutated.add(base)
            if handed and verdict is None:
                provisional[base] = handed
            continue
        if base in tier1:
            mutated.add(base)
            continue
        if method in RECEIVER_READONLY_WRITE_METHODS:
            continue  # df.to_csv reads the frame and writes a file
        if receiver_is_identity_coupled(receiver):
            # Any method on a Figure/Axes draws on it, whatever it returns
            # (ax.hist returns a data tuple). fig.savefig lands here too: a
            # Figure is never cached, so bumping it is harmless, and the edge
            # keeps the chart re-derived as a unit.
            mutated.add(base)
            continue
        if chain_is_pure(method, inner.get((base, method), frozenset())) or is_pandas_plot_call(method, receiver):
            continue
        if verdict is None:
            unknown.add(base)
        elif base in verdict:
            mutated.add(base)
    # A captured return routes its receiver only when it is a Figure/Axes or
    # an estimator being fitted, so a pure capture (m = df.mean()) caches.
    for base, method in assigned:
        if is_module_name(base, namespace, virtual_modules):
            continue
        receiver = namespace.get(base)
        if receiver_is_identity_coupled(receiver) or fits_its_receiver(method, receiver):
            mutated.add(base)
    # A call whose value is kept (``label = df.pop('label')``) or a ``del
    # df['label']`` changes its receiver all the same.
    mutated |= {base for base in changed if not is_module_name(base, namespace, virtual_modules)}
    mutated |= drawn_args
    if verdict is None:
        unknown_args = args
    else:
        mutated |= {name for name in args if name in verdict}
        unknown_args = frozenset()
    return ReceiverClasses(frozenset(mutated), frozenset(unknown), unknown_args, provisional)


def _receiver_root(node: ast.AST) -> str | None:
    """The name a method call's receiver is reached from: ``groups`` in
    ``groups.setdefault(k, []).append(v)``; None for an object a call of a
    bare name builds (``Path(p).write_text(x)``), which no name holds."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        return _receiver_root(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _receiver_root(node.func.value)
    return None
