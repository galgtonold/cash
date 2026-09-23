"""What a called function does to its arguments and to the globals it closes over.

Pure AST over the callee's source, which the caller supplies through a
``resolve_source`` callback (``name -> source or None``); nothing here reads a
namespace.
"""

from __future__ import annotations

import ast
import functools
import textwrap

from .ast_util import CallScope, called_names
from .mutations import _iter_store_targets, _MutationVisitor

__all__ = [
    "params_mutated_in_function",
    "standalone_call_arg_targets",
    "function_arg_mutations",
    "source_global_mutations",
    "callee_global_mutations",
    "stateful_self_functions",
    "partial_arg_mutations",
    "mutating_partials",
    "reduce_free_mutations",
    "stateful_closure_vars",
]


def _positional_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Ordered positional parameter names (posonly + normal), excluding *args."""
    return [a.arg for a in (*func.args.posonlyargs, *func.args.args)]


def _all_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every parameter name (posonly + normal + kwonly + *args + **kwargs)."""
    params = {
        a.arg
        for a in (
            *func.args.posonlyargs,
            *func.args.args,
            *func.args.kwonlyargs,
        )
    }
    if func.args.vararg:
        params.add(func.args.vararg.arg)
    if func.args.kwarg:
        params.add(func.args.kwarg.arg)
    return params


def _params_mutated_via_nested_calls(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    params: set[str],
    resolve_source,
    seen: frozenset[str],
) -> set[str]:
    """Params of *func* mutated only by being passed to another resolvable call.

    For each ``Name`` call in the body, resolve the callee, recursively find which
    of ITS params it mutates, and map those back to any of *func*'s params passed
    at the matching position / keyword.
    """
    out: set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        callee_name = node.func.id
        if callee_name in seen:
            continue  # recursion guard (mutual / self recursion)
        callee = _resolve_function_def(callee_name, resolve_source)
        if callee is None:
            continue
        callee_muts = params_mutated_in_function(callee, resolve_source, seen | {callee_name})
        if not callee_muts:
            continue
        pos_params = _positional_param_names(callee)
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name) and arg.id in params and i < len(pos_params) and pos_params[i] in callee_muts:
                out.add(arg.id)
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id in params and kw.arg in callee_muts:
                out.add(kw.value.id)
    return out


def _resolve_function_def(name, resolve_source):
    """Parse *name*'s source via *resolve_source* into a FunctionDef, or None."""
    if resolve_source is None:
        return None
    source = resolve_source(name)
    if not source:
        return None
    try:
        parsed = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return None
    if parsed.body and isinstance(parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return parsed.body[0]
    return None


def params_mutated_in_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    resolve_source=None,
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Parameter names a function body mutates IN PLACE.

     A parameter counts as mutated when the body performs an in-place mutation on
     it — subscript/attribute assignment, augmented assignment, a mutating method
     call, ``out=`` kwarg, or ``del`` — i.e. the same signals as
     :attr:`StatementAnalysis.all_mutated_vars`. Plain reassignment (``x = ...``)
     rebinds a local and does NOT mutate the caller's object, so it does not count.

     Used (with :func:`function_arg_mutations`) to attribute an argument mutation
     back to the caller's variable: ``def f(x): x.append(1)`` plus ``f(data)``
     means ``data`` is mutated in place, so it must reset on isolated re-run.

     When *resolve_source* is given (a ``name -> source`` lookup), the analysis is
     interprocedural: a parameter mutated only via a further resolvable call
     (``def outer(y): inner(y)`` where ``inner`` mutates its arg) is also detected
    . *seen* guards against mutual / self recursion. Without
     *resolve_source* the analysis is one level deep (the original
     behaviour).
    """
    params = _all_param_names(func)
    if not params:
        return frozenset()
    visitor = _MutationVisitor()
    for stmt in func.body:
        visitor.visit(stmt)
    mutated = {m.variable for m in visitor.mutations} & params
    if resolve_source is not None:
        mutated |= _params_mutated_via_nested_calls(func, params, resolve_source, seen)
    return frozenset(mutated)


def standalone_call_arg_targets(
    tree: ast.Module | None,
) -> frozenset[tuple[str, tuple[str | None, ...], tuple[tuple[str, str], ...]]]:
    """Top-level bare-``Expr`` calls to a NAME, with their variable arguments.

    Returns ``(func_name, positional, keywords)`` per call:

    * ``positional`` — a tuple with the variable name for each positional argument
      that is a bare ``Name``, or ``None`` for anything else (literal, expression,
      ``*args``) since only a tracked variable can be a reset target.
    * ``keywords`` — ``(param_name, arg_var)`` pairs for keyword arguments whose
      value is a bare ``Name``.

    Only bare-``Expr`` calls (result discarded) are returned: a call made purely
    for effect is the mutation pattern, whereas a pure call captures its result
    (``r = f(x)``). Method calls (``obj.m(x)``) are handled by the method-receiver
    path and excluded here.
    """
    if tree is None:
        return frozenset()
    out: set[tuple[str, tuple[str | None, ...], tuple[tuple[str, str], ...]]] = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name):
            continue
        positional = tuple(a.id if isinstance(a, ast.Name) else None for a in call.args)
        keywords = tuple(
            (kw.arg, kw.value.id) for kw in call.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)
        )
        out.add((call.func.id, positional, keywords))
    return frozenset(out)


def function_arg_mutations(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Caller variables mutated in place by being passed to a user-defined
    function that mutates the corresponding parameter.

    *resolve_source* maps a function name to its source string (or ``None`` if it
    is not a resolvable user-defined function — a builtin, C function, lambda, or
    unknown name). For each top-level bare-``Expr`` call the function body is
    parsed, its mutated parameters are found via :func:`params_mutated_in_function`
    (interprocedurally — a param mutated only through a further resolvable call is
    detected too), and each is mapped back to the call's positional /
    keyword argument variable.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for func_name, positional, keywords in standalone_call_arg_targets(tree):
        fdef = _resolve_function_def(func_name, resolve_source)
        if fdef is None:
            continue
        mutated_params = params_mutated_in_function(fdef, resolve_source, frozenset({func_name}))
        if not mutated_params:
            continue
        pos_params = _positional_param_names(fdef)
        for i, arg_var in enumerate(positional):
            if arg_var and i < len(pos_params) and pos_params[i] in mutated_params:
                out.add(arg_var)
        for param, arg_var in keywords:
            if param in mutated_params:
                out.add(arg_var)
    return frozenset(out)


def _free_vars_mutated_in_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Module-global / free variables a function body mutates in place.

    A name mutated in place (``items.append``, ``store[k]=``, ``g += 1`` under a
    ``global`` declaration) that is neither a parameter nor a plain local
    assignment is a free variable resolved from the enclosing / module scope —
    calling the function mutates that global. Parameter mutations are a
    separate job and are excluded; a name rebound locally (``acc = []`` then
    ``acc.append``) refers to the local and is excluded, UNLESS declared
    ``global`` / ``nonlocal``.
    """
    params = _all_param_names(func)
    global_decls: set[str] = set()
    local_assigned: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            global_decls.update(node.names)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        local_assigned.add(leaf.id)
    visitor = _MutationVisitor()
    for stmt in func.body:
        visitor.visit(stmt)
    mutated = {m.variable for m in visitor.mutations}
    local_assigned -= global_decls
    return frozenset(mutated - params - local_assigned)


@functools.lru_cache(maxsize=4096)
def source_global_mutations(source: str) -> frozenset[str]:
    """Globals the function defined by *source* mutates in place.

    The one per-callee answer to "which globals does calling this function
    change": the free variables its body mutates (see
    :func:`_free_vars_mutated_in_function`). Every engine asks this, whether it
    found the source through the user namespace, the notebook's cell text or a
    live function object, so they cannot disagree on what counts as a
    callee's write.

    Empty for anything that is not a single function definition; never raises.
    The verdict is purely syntactic (no namespace is consulted), so memoising
    on the source text is sound.
    """
    try:
        parsed = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, RecursionError):
        return frozenset()
    node = parsed.body[0] if parsed.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset()
    try:
        return _free_vars_mutated_in_function(node)
    except (ValueError, RecursionError):
        return frozenset()


def callee_global_mutations(
    tree: ast.AST | None,
    resolve_source,
    *,
    scope: CallScope = "all",
) -> frozenset[str]:
    """Globals mutated in place by the functions *tree* calls by name.

    *resolve_source* maps a called name to its source (or None when it is not a
    user function); each resolved callee contributes
    :func:`source_global_mutations`. *scope* picks the calls (see
    :data:`~cash.analysis.ast_util.CallScope`). A caller holding the namespace
    narrows the result with
    :func:`~cash.analysis.namespace_effects.capturable_globals`.

    Only the callee's own body counts: a global mutated by a helper the callee
    calls is not detected (the write is then skipped on a hit, as for any call
    cash cannot see into).

    The statement path asks with ``scope="no_control_bodies"``: a loop or
    branch is one unit to the upstream simulation and to the accumulator
    machinery, so a write in its body belongs to the control structure, not to
    one body statement (claiming it per statement makes the planner replay
    only the last writer). The upstream checker asks with ``scope="all"``: its
    idempotent-rerun reset acts per cell and must cover everything the cell
    writes, including through a loop.
    """
    out: set[str] = set()
    for name in called_names(tree, scope):
        try:
            source = resolve_source(name)
        except Exception:  # noqa: BLE001 - a resolver must never break analysis
            continue
        if source:
            out |= source_global_mutations(source)
    return frozenset(out)


_MUTABLE_LITERAL_CALLS = frozenset({"list", "dict", "set"})


def _is_mutable_default(node: ast.expr) -> bool:
    """True if *node* is a mutable literal default (``[]``, ``{}``, ``set()``)."""
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _MUTABLE_LITERAL_CALLS
        and not node.args
    )


_MEMOIZER_DECORATORS = frozenset({"lru_cache", "cache"})


def _is_memoizer_decorator(dec: ast.expr) -> bool:
    """True for a ``functools`` memoizer decorator — ``@lru_cache`` / ``@cache``,
    bare or called (``@lru_cache(maxsize=None)``), plain or dotted
    (``@functools.lru_cache``)."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Name):
        return node.id in _MEMOIZER_DECORATORS
    if isinstance(node, ast.Attribute):
        return node.attr in _MEMOIZER_DECORATORS
    return False


def _function_mutates_own_object(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if calling *func* mutates state carried on the function OBJECT itself
    (part B), which persists across calls and must be reset by re-running
    the ``def``:

    * a **mutable default argument** the body mutates in place
      (``def collect(x, acc=[]): acc.append(x)``);
    * an assignment to a **function attribute**
      (``def tick(): tick.count = getattr(tick, 'count', 0) + 1``);
    * a **functools memoizer** (``@lru_cache`` / ``@cache``) — the cache persists
      across calls, so a re-run reuses memoised results and its body's
      side effects (a free-var append) no longer fire; re-running the ``def``
      recreates an empty cache.
    """
    if any(_is_memoizer_decorator(d) for d in func.decorator_list):
        return True
    fname = func.name
    for node in ast.walk(func):
        target = node.target if isinstance(node, ast.AugAssign) else None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == fname:
                    return True
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == fname:
            return True

    mutated = params_mutated_in_function(func)
    if mutated:
        pos = list(func.args.posonlyargs) + list(func.args.args)
        defs = func.args.defaults
        if defs:
            for param, default in zip(pos[-len(defs) :], defs):
                if param.arg in mutated and _is_mutable_default(default):
                    return True
        for kw, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
            if default is not None and kw.arg in mutated and _is_mutable_default(default):
                return True
    return False


def stateful_self_functions(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Called functions that carry mutable state on their own object (B).

    For each function called in the cell (captured or bare) whose source
    resolves, return its name if calling it mutates state on the function object
    (a mutated mutable default arg, or a function-attribute assignment). The
    checker force-resets these so their ``def`` re-runs and recreates fresh state
    on an isolated re-run, instead of the default / attribute accumulating.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in called_names(tree):
        fdef = _resolve_function_def(name, resolve_source)
        if fdef is not None and _function_mutates_own_object(fdef):
            out.add(name)
    return frozenset(out)


def partial_arg_mutations(tree: ast.Module | None, resolve_partial, resolve_source) -> frozenset[str]:
    """Vars mutated through a called ``functools.partial`` binding.

    *resolve_partial* maps a name to ``(target_func_name, [bound_arg_vars])`` for a
    ``p = partial(f, x, y)`` binding, or ``None``. For each partial called in the
    cell, the target ``f``'s in-place param mutations are mapped to the bound
    positional args (``partial(push, shared)`` + ``p('a')`` mutates ``shared``
    because ``push`` mutates its first param), and ``f``'s free/global mutations
    are attributed too (``partial(tick, 1)`` where ``tick`` mutates ``counter``).
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in called_names(tree):
        binding = resolve_partial(name)
        if binding is None:
            continue
        f_name, bound_args = binding
        fdef = _resolve_function_def(f_name, resolve_source)
        if fdef is None:
            continue
        mutated_params = params_mutated_in_function(fdef)
        pos_params = _positional_param_names(fdef)
        for i, arg in enumerate(bound_args):
            if arg and i < len(pos_params) and pos_params[i] in mutated_params:
                out.add(arg)
        out |= _free_vars_mutated_in_function(fdef)
    return frozenset(out)


def mutating_partials(tree: ast.Module | None, resolve_partial, resolve_source) -> frozenset[str]:
    """Partial vars called in the cell that bind a MUTATED positional arg.

    ``p = partial(push, shared)`` captures the ``shared`` LIST OBJECT, so resetting
    the ``shared`` name to a fresh list does not help — ``p`` still appends to the
    old object. These partials must be re-created (their ``def`` re-run) so they
    re-bind to the reset arg. A partial whose target mutates only a FREE/global var
    is NOT included: a global is resolved dynamically, so resetting it suffices.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in called_names(tree):
        binding = resolve_partial(name)
        if binding is None:
            continue
        f_name, bound_args = binding
        fdef = _resolve_function_def(f_name, resolve_source)
        if fdef is None:
            continue
        mutated_params = params_mutated_in_function(fdef)
        pos_params = _positional_param_names(fdef)
        for i, arg in enumerate(bound_args):
            if arg and i < len(pos_params) and pos_params[i] in mutated_params:
                out.add(name)
                break
    return frozenset(out)


def reduce_free_mutations(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Free/global vars mutated by a function passed to ``functools.reduce``.

    ``reduce(combine, [1, 2, 3], 0)`` invokes ``combine`` once per element; if
    ``combine`` mutates a free variable (``log.append(b)``) that mutation happens,
    so the free var must reset on isolated re-run.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_reduce = (isinstance(fn, ast.Name) and fn.id == "reduce") or (
            isinstance(fn, ast.Attribute) and fn.attr == "reduce"
        )
        if is_reduce and node.args and isinstance(node.args[0], ast.Name):
            fdef = _resolve_function_def(node.args[0].id, resolve_source)
            if fdef is not None:
                out |= _free_vars_mutated_in_function(fdef)
    return frozenset(out)


def _factory_body_scope(factory: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names bound in the factory's OWN scope: its params + names assigned at its
    direct body level (the locals a returned closure captures)."""
    scope = _all_param_names(factory)
    for stmt in factory.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        scope.add(leaf.id)
        elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and isinstance(stmt.target, ast.Name):
            scope.add(stmt.target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.add(stmt.name)
    return scope


def _factory_returns_stateful_closure(factory: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if *factory* defines an inner function that mutates a variable
    captured from the factory's own scope (a ``nonlocal`` cell or a factory-local
    container). Such a closure carries state that persists across calls of the
    returned function and is only reset by re-running the factory (B).
    """
    factory_scope = _factory_body_scope(factory)
    for node in ast.walk(factory):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not factory:
            if _free_vars_mutated_in_function(node) & factory_scope:
                return True
    return False


def stateful_closure_vars(tree: ast.Module | None, resolve_var_factory) -> frozenset[str]:
    """Closure variables called in the cell that carry mutable captured state
    (B). *resolve_var_factory* maps a name to the ``FunctionDef`` of the
    factory it was assigned from (``c = make_counter()`` → ``make_counter``'s
    def), or ``None``. A name is flagged when its factory returns a closure that
    mutates factory-local state, so the checker force-resets it and its producer
    (``c = make_counter()``) re-runs to recreate the fresh closure.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in called_names(tree):
        factory = resolve_var_factory(name)
        if factory is not None and _factory_returns_stateful_closure(factory):
            out.add(name)
    return frozenset(out)
