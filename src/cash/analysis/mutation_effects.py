"""What running a cell or a statement changes, as one typed result.

The upstream checker needs to know everything a cell writes, split by the
channel an isolated re-run resets it through (:class:`CellEffects`); the
statement processor and the upstream simulator need to know what one statement
reads and writes (:class:`StatementEffects`). Each used to chain the analyses
in ``cacheability`` by hand. They now ask :func:`cell_effects` and
:func:`statement_effects`, so a new mutation channel is added in one place.

Functions called by name are resolved through a ``name -> source`` callable
the caller supplies: the runtime reads the live namespace, the simulation and
the checker read the notebook's cell text. Only the resolver differs between
engines; the rules do not.
"""

from __future__ import annotations

import ast
import functools
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .annotations import extract_annotations_for_statements
from .ast_util import called_names
from .cacheability import (
    RECEIVER_READONLY_WRITE_METHODS,
    alias_mutation_sources,
    aliased_sources,
    analyze_statement,
    assigned_method_call_receivers,
    callee_global_mutations,
    chain_is_pure,
    crossref_reassigned_vars,
    fits_its_receiver,
    function_arg_mutations,
    is_pandas_plot_call,
    module_setting_receivers,
    mutating_partials,
    object_protocol_mutations,
    partial_arg_mutations,
    reduce_free_mutations,
    selfref_inplace_write_vars,
    standalone_call_arg_targets,
    standalone_method_call_inner_methods,
    standalone_method_call_receivers,
    standalone_method_mutation_receivers,
    stateful_closure_vars,
    stateful_self_functions,
    subscript_view_bindings,
    top_level_call_argument_bases,
)
from .cacheability_decision import receiver_is_identity_coupled
from .code_analyzer import CodeAnalyzer

__all__ = [
    "CellEffects",
    "NotebookSources",
    "ReceiverClasses",
    "StatementEffects",
    "cell_effects",
    "classify_receivers",
    "drawn_on_arguments",
    "is_module_name",
    "nocache_written_vars",
    "statement_effects",
]

SourceResolver = Callable[[str], "str | None"]


# ---------------------------------------------------------------------------
# Notebook-wide definitions, read from cell text
# ---------------------------------------------------------------------------


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
        bodies = []
        for code in (*self.cells, self._current_cell):
            try:
                bodies.append(ast.parse(code).body)
            except (SyntaxError, ValueError):
                continue
        return bodies

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
        for node in self._top_level(kinds):
            try:
                sources[node.name] = ast.unparse(node)
            except (ValueError, AttributeError):
                continue
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


def nocache_written_vars(cell_code: str) -> frozenset[str]:
    """Variables written by a ``# @cash: no-cache`` statement in *cell_code*.

    ``no-cache`` means "always run fresh", so a self-modifying variable under
    it (``counter = counter + 1``) is meant to accumulate on a re-run, not be
    restored to its input.
    """
    try:
        annotations = extract_annotations_for_statements(cell_code)
        if not annotations:
            return frozenset()
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
        return _cell_effects(cell_code, sources, namespace)
    except (SyntaxError, ValueError):
        return CellEffects()


def _cell_effects(cell_code: str, sources: NotebookSources, namespace: Mapping[str, Any]) -> CellEffects:
    _, outputs = CodeAnalyzer.analyze_code_block(cell_code)
    reassigned = CodeAnalyzer.reassigned_names(cell_code)
    tree = ast.parse(cell_code)
    calls_something = bool(called_names(tree))
    # Globals a callee mutates count as mutated here, exactly like an inline
    # mutation, so the reset covers them. They are deliberately NOT outputs:
    # the checker reads `outputs` as "written by the cell, not an input to
    # restore", which would disable the very reset that makes the statement's
    # key converge.
    mutated = set(
        analyze_statement(
            cell_code, None, resolve_source=sources.functions.get if calls_something else None
        ).all_mutated_vars
    )
    nocache = nocache_written_vars(cell_code)
    reassigned -= nocache
    mutated -= nocache
    # Receivers of a bare method call (``b.items.append``): such no-output
    # statements skip the per-statement cache, so a lineage-carrying receiver
    # would accumulate on an isolated re-run. Method receivers only, so
    # ``df['col'] = ...`` keeps its per-statement cache.
    method_receivers = set(standalone_method_mutation_receivers(tree)) - nocache
    # Self-referential subscript/attr writes (``df['a'] = df['a'] * 2``,
    # ``df.iloc[i, j] += x``) are not idempotent. A new column read from
    # other columns is not self-referential and keeps its cache.
    selfref = set(selfref_inplace_write_vars(tree)) - nocache
    stateful: set[str] = set()
    hidden: set[str] = set()

    # A variable passed to a helper that mutates that parameter
    # (``def add(d): d.append(x)`` + ``add(data)``) is reset like a receiver.
    if standalone_call_arg_targets(tree):
        arg_muts = function_arg_mutations(tree, sources.functions.get) - nocache
        mutated |= arg_muts
        method_receivers |= arg_muts

    if calls_something:
        functions = sources.functions.get
        # A global a called function mutates (``def bump(): global g; g += 1``)
        # joins the inputs so its producer's base is restored. Every call
        # counts, not only a bare-``Expr`` one: the statement keys on the
        # global's pre-state, so without the reset the value it produced would
        # be the next run's key -- a cell that re-executes and accumulates
        # forever.
        global_muts = callee_global_mutations(tree, functions) - nocache
        mutated |= global_muts
        hidden |= global_muts
        # State on the function object itself (a mutated mutable default, a
        # function attribute, a memoizer): re-run its ``def``.
        stateful |= set(stateful_self_functions(tree, functions)) - nocache
        # A closure (``c = make_counter()``) whose factory's inner function
        # mutates factory-local state: re-run the factory call.
        stateful |= set(stateful_closure_vars(tree, sources.factory_def)) - nocache
        # Mutation through functools.partial or a functools.reduce callback.
        partials = sources.partial_bindings.get
        partial_muts = (
            partial_arg_mutations(tree, partials, functions) | reduce_free_mutations(tree, functions)
        ) - nocache
        mutated |= partial_muts
        hidden |= partial_muts
        # A partial that bound a mutated argument holds the argument's object:
        # re-bind it too.
        stateful |= set(mutating_partials(tree, partials, functions)) - nocache

    op = _object_protocol_effects(cell_code, tree, sources, nocache)
    if op is not None:
        op_free, op_receivers, op_class_defs, op_init_free = op
        mutated |= op_free | op_receivers | op_init_free
        hidden |= op_free | op_init_free
        method_receivers |= op_receivers
        stateful |= op_class_defs

    # ``y = x`` shares x's object, so ``y.append`` also mutates x. The selfref
    # set is column-scoped, so an aliased NEW-column write keeps its cache.
    mutated |= alias_mutation_sources(tree) - nocache
    selfref |= aliased_sources(tree, selfref) - nocache
    method_receivers |= aliased_sources(tree, method_receivers) - nocache
    # ``v = arr[slice]`` on a numpy array is a view: mutating v mutates arr.
    # A list slice is a copy, hence the live-value check.
    view_bindings = subscript_view_bindings(tree)
    if view_bindings:
        mutated_here = analyze_statement(cell_code, None).all_mutated_vars
        mutated |= {
            base
            for alias, base in view_bindings.items()
            if alias in mutated_here and _is_live_ndarray(namespace.get(base))
        } - nocache
    # ``a, b = b, a`` reads its own pre-cell value but, re-run alone, holds the
    # swapped output: lineage-invisible, so reset it by rule.
    crossref = crossref_reassigned_vars(tree) - nocache

    return CellEffects(
        outputs=frozenset(outputs),
        reassigned=frozenset(reassigned),
        mutated=frozenset(mutated),
        method_receivers=frozenset(method_receivers),
        selfref=frozenset(selfref),
        crossref_reassigned=frozenset(crossref),
        stateful_funcs=frozenset(stateful),
        nocache=nocache,
        hidden_inputs=frozenset(hidden),
    )


def _object_protocol_effects(
    cell_code: str, tree: ast.Module, sources: NotebookSources, nocache: frozenset[str]
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]] | None:
    """Hidden state reached through the object protocol, per reset channel:
    ``(free_vars, receivers, class_defs, init_subclass_free_vars)``.

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

    resets = object_protocol_mutations(
        tree,
        sources.classes.get,
        sources.instance_class,
        sources.functions.get,
        op_var_factory,
        decorated_class=sources.decorated_class,
    )
    free = resets.free_vars - nocache
    receivers = resets.receivers - nocache
    class_defs = resets.class_defs - nocache
    # An ``__init_subclass__`` registry hides its mutation behind class
    # creation, so the simulator's content-base guard cannot see a sibling
    # subclass cell also appending to it; it gets the cross-cell check below.
    init_free = resets.init_subclass_free_vars - nocache
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
                other = object_protocol_mutations(
                    ast.parse(code),
                    sources.classes.get,
                    sources.instance_class,
                    sources.functions.get,
                    op_var_factory,
                )
            except (SyntaxError, ValueError):
                continue
            other_receivers |= other.receivers
            other_class_defs |= other.class_defs
            other_init_free |= other.init_subclass_free_vars
        receivers -= other_receivers
        class_defs -= other_class_defs
        init_free -= other_init_free
    return free, receivers, class_defs, init_free


# ---------------------------------------------------------------------------
# Statement effects: what the runtime and the simulation key and bump
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatementEffects:
    """What one statement reads and writes, as both engines must see it."""

    #: Names the statement reads.
    inputs: frozenset[str]
    #: Names it binds, plus the globals a callee mutates in place.
    outputs: frozenset[str]
    #: Notebook globals a callee mutates in place. Empty for a statement in a
    #: control-structure body: the loop or branch owns its body's writes.
    callee_globals: frozenset[str]
    #: Variables handed to a bare call of a user function that mutates the
    #: matching parameter in place (``def add(d): d.append(x)`` + ``add(data)``).
    arg_mutations: frozenset[str]


def is_module_name(name: str, namespace: Mapping[str, Any], virtual_modules: Iterable[str] = ()) -> bool:
    """Whether *name* is bound to a module: live in *namespace*, or, when it is
    not bound there yet, among the *virtual_modules* a simulation bound."""
    if isinstance(namespace.get(name), types.ModuleType):
        return True
    return name not in namespace and name in virtual_modules


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
    magic-tolerant parse, and no callee is looked into. *virtual_modules* are
    names the simulation bound to modules that the kernel does not hold yet.
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=tree, resolve_source=resolve_source, user_ns=namespace)
    callee_globals: frozenset[str] = frozenset()
    if not control_body:
        callee_globals = callee_global_mutations(tree, resolve_source, scope="no_control_bodies", namespace=namespace)
    arg_mutations: frozenset[str] = frozenset()
    try:
        if standalone_call_arg_targets(tree):
            arg_mutations = frozenset(
                v
                for v in function_arg_mutations(tree, resolve_source)
                if not is_module_name(v, namespace, virtual_modules)
            )
    except (SyntaxError, ValueError, RecursionError):
        arg_mutations = frozenset()
    return StatementEffects(frozenset(inputs), frozenset(outputs), callee_globals, arg_mutations)


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
    #: Bare-call arguments no verdict covers. The runtime fingerprints them;
    #: the simulation leaves them alone, since treating every ``print(df)``
    #: as a change would bump ``df`` for every reader.
    unknown_args: frozenset[str] = frozenset()


def drawn_on_arguments(tree: ast.Module | None, namespace: Mapping[str, Any]) -> frozenset[str]:
    """Figures and Axes handed to a call, which draws on them: by a method
    (``df.plot(ax=ax)``) or by a plain function (``forest(axes[0], df)``)."""
    return frozenset(
        name for name in top_level_call_argument_bases(tree) if receiver_is_identity_coupled(namespace.get(name))
    )


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
    bare-call arguments the engine considers (see ``bare_call_arguments``).
    A module is never a receiver: ``pd.set_option(...)`` is a module
    function call, counted only as a change to a setting the module keeps.
    """
    candidates = standalone_method_call_receivers(tree)
    # A captured return (``counts, bins, _ = ax.hist(...)``) is an assignment,
    # so it is not a bare-``Expr`` candidate; it must not hit the early return.
    assigned = assigned_method_call_receivers(tree)
    drawn_args = drawn_on_arguments(tree, namespace)
    args = frozenset(arguments) - drawn_args
    if not candidates and not assigned and not drawn_args and not args:
        return ReceiverClasses()
    tier1 = standalone_method_mutation_receivers(tree)
    inner = standalone_method_call_inner_methods(tree)
    settings = module_setting_receivers(tree)
    verdict = load_verdict()
    mutated: set[str] = set()
    unknown: set[str] = set()
    for base, method in candidates:
        receiver = namespace.get(base)
        if is_module_name(base, namespace, virtual_modules):
            if base in settings:  # pd.set_option, plt.rcParams.update
                mutated.add(base)
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
    mutated |= drawn_args
    if verdict is None:
        unknown_args = args
    else:
        mutated |= {name for name in args if name in verdict}
        unknown_args = frozenset()
    return ReceiverClasses(frozenset(mutated), frozenset(unknown), unknown_args)
