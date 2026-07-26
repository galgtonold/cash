from __future__ import annotations

"""Sub-expression caching: selecting call nodes to cache independently (CAS-243).

**The problem.** The unit of caching is the statement. When a statement is a
cheap wrapper around an expensive call, that unit is wrong in both directions::

    out.append(compute(x))   # skip-cached (the append is a mutation) -> no reuse, ever
    s += compute(x)          # cached, but keyed on the running prefix -> a reorder
                             # re-runs everything after the first change

In both, the expensive thing is ``compute(x)`` and the cheap thing is the
wrapper. Caching the wrapper either fails outright or attaches an irrelevant
dependency; caching the call is right in both.

**The rule.** A call is eligible when its free variables do **not** include the
statement's assignment / mutation target. If the call reads the target it *is*
the fold, and no order-independent value can be extracted from it.

That single rule does more work than it appears to. In ``out.append(f(x))`` the
append is itself a ``Call`` whose func reads ``out`` — the target — so the same
rule that admits ``f(x)`` excludes the mutation. There is no special case for
"don't cache the mutation itself".

**Scope of this module.** *Structural* eligibility only, from the AST. Whether a
particular callee is worth intercepting at runtime — already ``@cash.cache``-d,
a builtin, not a function at all — is an object-level question answered where
the live object is in hand, not here.
"""

import ast
import copy
import types

__all__ = ["eligible_call_nodes", "wrap_eligible_calls", "CallCache", "HELPER_NAME"]

#: Name bound in ``user_ns`` that resolves a callee to its cached counterpart.
#: Dunder-prefixed so it cannot collide with a user's own names.
HELPER_NAME = "__cash_call__"


class CallCache:
    """Resolves a callee to the thing that should actually be called.

    The AST decides *structural* eligibility; this is the object-level gate,
    which needs the live callable in hand. Three outcomes:

    - a plain Python function -> its ``@cash.cache`` counterpart,
    - a function already decorated -> itself. It is already on this path;
      wrapping again would mint a second key for the same work and split its
      hits across two entries.
    - anything else -> itself. Builtins (``len``, ``print``) and types
      (``str``, ``range``) are too cheap to be worth a key and would make a
      hot loop pay for one per iteration.

    Bound methods are passed through in this first cut. They are callables like
    any other and nothing here prevents caching them later, but keying a method
    means keying its receiver too, which is a separate decision.

    **This must never be why user code breaks.** Anything unrecognised, and any
    failure to build a wrapper, hands the original callable back.
    """

    def __init__(self, cash_instance):
        self._cash = cash_instance
        # Keyed by id() and pinned by the value tuple: a function's id can be
        # reused after garbage collection, so the original is held alongside
        # the wrapper to keep it alive and to detect a recycled id.
        self._wrappers: dict[int, tuple[types.FunctionType, object]] = {}

    def resolve(self, fn):
        """Return *fn* or a cached counterpart. Never raises."""
        if not isinstance(fn, types.FunctionType):
            return fn
        if getattr(fn, '_cash_cached', False):
            return fn

        entry = self._wrappers.get(id(fn))
        if entry is not None and entry[0] is fn:
            return entry[1]

        try:
            wrapper = self._cash.cache(fn)
        except Exception:  # noqa: BLE001 - a caching wrapper is never worth an error
            return fn
        self._wrappers[id(fn)] = (fn, wrapper)
        return wrapper


def wrap_eligible_calls(tree: ast.Module) -> tuple[ast.Module, int]:
    """Return ``(rewritten_copy, n_wrapped)``; *tree* is left untouched.

    Each eligible call has its **callee expression** wrapped::

        compute(x)  ->  __cash_call__(compute)(x)

    The argument list is not rewritten at all, so ``*args``/``**kwargs``,
    keyword arguments and evaluation order need no special handling — and,
    critically, the call stays exactly where it was in the expression. A
    short-circuited ``g()`` in ``f() or g()`` is still only reached when ``f()``
    is falsy; hoisting it into a temporary would have run it unconditionally.

    The copy matters: the caller keeps using the original tree for analysis and
    cache keying, and rewriting in place would desync the runtime's source from
    the upstream simulator's.
    """
    new_tree = copy.deepcopy(tree)
    count = 0
    for stmt in new_tree.body:
        for call in eligible_call_nodes(stmt):
            call.func = ast.Call(
                func=ast.Name(id=HELPER_NAME, ctx=ast.Load()),
                args=[call.func],
                keywords=[],
            )
            count += 1
    if count:
        ast.fix_missing_locations(new_tree)
    return new_tree, count


#: Statement shapes the free-variable rule is sound for. Everything else --
#: compound statements and definitions -- is declined; see the note in
#: :func:`eligible_call_nodes`.
_SIMPLE_STATEMENTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr)


def eligible_call_nodes(stmt: ast.stmt) -> list[ast.Call]:
    """Return the calls in *stmt* that may be cached independently of it.

    Outermost-first, in source order, and never nested: once a call is
    accepted its subtree is not searched, because intercepting the outer call
    already covers everything inside it. Returning both would mint two cache
    entries for one piece of work.

    **Only simple statements are searched, and that is a safety rule rather
    than a simplification.** Cash can execute a loop as a single unit, in which
    case the node handed here is the ``ast.For`` itself — which has no
    assignment target, so the free-variable rule would exclude nothing and every
    call in the body would look eligible, including the side-effecting one the
    loop exists to perform::

        for x in xs:
            log_it(x)        # caching this would skip the log on every re-run

    The rule is only sound against a target, and per-iteration decomposition
    already hands each body statement here separately, with its own. The same
    reasoning declines ``def``/``class`` bodies: they run later, under their own
    statement.
    """
    if not isinstance(stmt, _SIMPLE_STATEMENTS):
        return []
    targets = _target_names(stmt)
    found: list[ast.Call] = []
    for root in _search_roots(stmt):
        _collect(root, targets, found)
    return found


def _search_roots(stmt: ast.stmt) -> list[ast.AST]:
    """The expression subtrees worth searching for cacheable calls.

    For an expression statement the outermost call is the *effect* the
    statement exists for — ``out.append(...)``, ``print(...)``, a draw on an
    Axes. It must never be intercepted (restoring it would skip the effect), so
    the search starts below it, at its arguments and its callee expression.
    """
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return list(ast.iter_child_nodes(stmt.value))
    return [stmt]


def _collect(node: ast.AST, targets: set[str], found: list[ast.Call]) -> None:
    if isinstance(node, ast.Call) and not (_names_read(node) & targets):
        found.append(node)
        return  # accepted -- do not search inside it
    for child in ast.iter_child_nodes(node):
        _collect(child, targets, found)


def _names_read(node: ast.AST) -> set[str]:
    """Every bare name appearing anywhere under *node*.

    Deliberately not scope-aware: ``s.total`` and ``s[0]`` must both count as
    reading ``s``, or a fold that touches the accumulator through an attribute
    would look independent of it and get cached against a stale value.
    """
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _target_names(stmt: ast.stmt) -> set[str]:
    """The names *stmt* assigns to or mutates in place.

    For an expression statement that is a method call, the receiver is the
    mutation target: ``out.append(...)`` targets ``out``. A plain function call
    (``print(...)``) has no receiver and so no target, which is correct — the
    call itself is already excluded as the statement's effect.
    """
    if isinstance(stmt, ast.Assign):
        return set().union(*(_base_names(t) for t in stmt.targets)) if stmt.targets else set()
    if isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        return _base_names(stmt.target)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        if isinstance(func, ast.Attribute):
            return _base_names(func.value)
    return set()


def _base_names(node: ast.AST) -> set[str]:
    """The root name(s) a target expression is rooted at.

    ``prices[t]`` -> ``prices``; ``obj.attr`` -> ``obj``; ``a, b`` -> both.
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        return _base_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_base_names(e) for e in node.elts)) if node.elts else set()
    return set()
