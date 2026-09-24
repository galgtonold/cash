"""Sub-expression caching: selecting call nodes to cache independently.

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

**Caveat for large loops.** The ``out.append(compute(x))`` example above only
reaches this module when the loop is decomposed per-iteration.
``single_unit_policy.should_run_as_single_unit`` routes a large-enough loop
to the single-unit fast path instead, gated by
``MIN_ITERATIONS_FOR_SINGLE_UNIT``, ``PER_STMT_OVERHEAD_SEC``, and
``MIN_OVERHEAD_SEC`` in that module. Calls inside a single-unit loop are
searched per body statement (``_eligible_calls_in_loop``) and cached only when
keyed on the values they receive. For the precise threshold, see
``docs/known-limitations.md``'s "A long for-append loop can stop caching"
section, whose numbers are pinned to those constants by a claim-anchor test
(``tests/docs/test_claim_anchors.py``).
"""

from __future__ import annotations

import ast
import builtins
import copy
import inspect
import types
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .call_key import global_names_reached
from .lineage_formula import is_cash_instrumentation

__all__ = [
    "eligible_call_nodes",
    "wrap_eligible_calls",
    "CallSite",
    "HELPER_NAME",
]

#: Name bound in ``user_ns`` that resolves a callee to its cached counterpart.
#: Dunder-prefixed so it cannot collide with a user's own names.
HELPER_NAME = "__cash_call__"


@dataclass(frozen=True)
class CallSite:
    """Everything the runtime needs about one intercepted call, computed once.

    Built at rewrite time so the thunk never re-parses or re-unparses on the
    hot path: a loop body calling this 10,000 times would otherwise pay an
    ``ast.unparse`` per iteration.

    ``occurrence_index`` counts *identical sources within the cell*, matching
    the statement path's rule (``compute_cache_key``'s ``occ`` component). Two
    spellings of the same call are different sources and each start at 0.
    """

    source: str
    free_names: frozenset[str]
    occurrence_index: int
    #: Positions (in ``(*args, *kwargs.values())`` order) of arguments that are
    #: NOT a bare ``ast.Name``. Those are the ones whose *evaluated value* must
    #: be hashed into the key, because their value is not a function of any
    #: name's lineage -- ``compute(next(it))`` being the case that makes this
    #: a correctness requirement rather than a tuning knob.
    computed_arg_positions: tuple[int, ...] = ()
    #: True when the call uses ``*args``/``**kwargs`` unpacking. The runtime
    #: half (``CallUnit``) must refuse to key such a call at all: the live
    #: arity it sees (the flattened ``(*args, *kwargs.values())`` it is
    #: actually called with) can differ from ``len(computed_arg_positions)``,
    #: which is a STATIC count from the AST -- hashing only the positions that
    #: count predicts would silently ignore any unpacked elements beyond it.
    has_unpacking: bool = False
    #: ``ast.unparse`` of the statement that CONTAINS this call -- computed
    #: once per enclosing statement, before any call inside it is rewritten.
    #:``call_cache_key``'s base key is built from the call's OWN
    #: source and free names, which says nothing about which statement the
    #: call sits in: two different statements whose call text and free names
    #: happen to agree (``vals[step] = fetch_next(conn)`` in one loop,
    #: ``other[step] = fetch_next(conn)`` in another, same loop items) shared
    #: one base key and the second was served the first's cached values --
    #: wrong on the first run, no pre-existing cache required. Folding this in
    #: fixes it.
    #:
    #: **Why ``ast.unparse``, not the raw statement source string.** A
    #: control-structure body statement arrives with a
    #: ``# __iteration_context__: <hash>`` comment PREPENDED to its source
    #: text (``for_handler.py``), and that context hash is derived in part
    #: from ``__iterable_lineage__`` -- the whole iterable's lineage, which
    #: changes for every iteration on a reorder. If that raw text
    #: reached this field, reordering a loop's items would change every
    #: iteration's statement identity and re-run the whole tail -- the exact
    #: bug ``loop_vars`` (not the iteration-context comment) exists to fix
    #: correctly. ``ast.unparse`` re-serialises the parsed AST, which never
    #: contained the comment in the first place (comments are not AST nodes),
    #: so it is stable across iterations and across reorders while still
    #: distinguishing genuinely different statements. This is also the same
    #: comment-free convention the upstream simulator already uses to key a
    #: statement (``upstream/virtual_lineage.py``'s ``ast.unparse(node)``).
    #:
    #: Default ``""`` (not folded into the key at all -- see
    #: ``call_cache_key``) so every pre-existing direct ``CallSite``
    #: construction, including every unit test that predates this field,
    #: keeps building the exact key it built before. A caching optimisation
    #: must never be why user code fails: if the enclosing statement's
    #: identity cannot be obtained (see :func:`wrap_eligible_calls`'s
    #: ``try/except`` around the ``ast.unparse`` call), this degrades to
    #: today's behaviour -- the pre-existing collision risk -- rather than
    #: raising.
    stmt_identity: str = ""
    #: Positions of arguments that read a name bound by an enclosing
    #: comprehension or lambda -- ``slow(v)`` in ``{k: slow(v) for k, v in
    #: d.items()}``. Such a name is local to the comprehension: resolving it by
    #: name, the key found either nothing or an unrelated global of the same
    #: name, so every element got ONE key and the second call was served the
    #: first's result -- on the first run with a global ``v`` around.
    #: These
    #: are also in ``computed_arg_positions``, and are hashed in full, never
    #: sampled, for the reason a loop variable is (see
    #: ``call_key._loop_var_digest``): the argument's value is then all that
    #: tells the elements apart. That holds for an argument computed from the
    #: name too -- ``fit_score(make_features(cleaned[mid], W))``: a
    #: frame's sample is its shape, dtypes and first five rows, and rolling
    #: features all begin with the same empty rows.
    local_arg_positions: tuple[int, ...] = ()
    #: Free names the call reads only inside computed arguments -- ``cleaned``
    #: and ``make_features`` in ``fit_score(make_features(cleaned[mid], W))``.
    #: What they contribute is the argument's value, which the key can hash
    #: instead of their lineage (see ``call_key.call_cache_key``'s
    #: *by_content*): a change to ``cleaned`` that leaves this element's
    #: features as they were then keeps the call's result.
    content_names: frozenset[str] = frozenset()
    #: ``(name, position)`` of each argument passed as a bare name that the
    #: call reads nowhere else -- ``PARAMS`` and ``cutoff`` in
    #: ``fit_series(g, PARAMS, cutoff)``. Under content keying the key may
    #: hold such a value instead of the name's lineage (see
    #: ``call_key.call_cache_key``'s *name_digests*).
    name_arg_positions: tuple[tuple[str, int], ...] = ()
    #: The call as its content key sees it: each computed argument replaced by
    #: a placeholder for its position (``fit_score(_arg0)``). Under content
    #: keying that argument's value is in the key, so how it was spelled is
    #: not: the sweep's ``make_features(cleaned[mid], W)`` and the pick's
    #: ``make_features(cleaned[mid], BEST_W)`` hand ``fit_score`` the same
    #: features, and keyed on the text the pick re-fitted all of them.
    #:Under unpacking, the callee alone: the key then
    #: holds every value received, with its keyword.
    content_source: str = ""
    #: Inside a loop cached as one unit (see ``_eligible_calls_in_loop``): no
    #: name read there has a lineage the key can trust, so the call is cached
    #: only when keyed on what it receives, and otherwise runs plain.
    in_loop_unit: bool = False


def interceptable(fn) -> bool:
    """Whether a callee is one :meth:`CallCache.resolve` wraps.

    Only a plain Python function: builtins, classes and bound methods are left
    alone (see "Bound methods are deliberately not intercepted" in the docs).
    Also not one that is already ``@cash.cache``-d, nor ``@cash.stateful`` --
    THE documented way to say "never cache this"; skipping that check cached a
    stateful callee and returned a stale value on the FIRST run (``[1, 1]``
    where plain Python gives ``[1, 2]``), because the second call in a loop hit
    the entry the first had just written. Nor cash's own instrumentation:
    the file tracker wraps ``pd.read_csv`` and the other readers that raise
    no audit event in tracking wrappers, which are plain functions; wrapping one means trying to
    cache a file handle. Nor IPython's ``open``: the kernel binds
    ``user_ns['open']`` to a plain-function wrapper of ``io.open``, so a bare
    ``open(p)`` in a cell was intercepted like a user function. Its 3 ms cost
    floor kept it uncached on an idle machine; under load the call crossed it,
    the handle was stored, and the next run was handed the one a reader had
    already drained. ``is_cash_instrumentation`` is the test the key already
    applies to both.
    """
    return (
        isinstance(fn, types.FunctionType)
        and not getattr(fn, "_cash_cached", False)
        and not getattr(fn, "_cash_stateful", False)
        and not is_cash_instrumentation(fn)
    )


_NOT_FOUND = object()


def _static_callee(node: ast.AST, namespace) -> object:
    """The object a callee expression names, found without calling anything.

    A bare name through *namespace* and the builtins; an attribute only through
    a module or a class (a static lookup on a class, so no descriptor or
    property runs). Anything else -- an instance's method, a subscript, a call
    result -- is ``_NOT_FOUND``.
    """

    if isinstance(node, ast.Name):
        if node.id in namespace:
            return namespace[node.id]
        return getattr(builtins, node.id, _NOT_FOUND)
    if isinstance(node, ast.Attribute):
        base = _static_callee(node.value, namespace)
        try:
            if isinstance(base, types.ModuleType):
                return getattr(base, node.attr, _NOT_FOUND)
            if isinstance(base, type):
                found = inspect.getattr_static(base, node.attr, _NOT_FOUND)
                return found.__func__ if isinstance(found, staticmethod) else found
        except Exception:  # noqa: BLE001 - a lookup is never worth an error
            return _NOT_FOUND
    return _NOT_FOUND


def wrap_eligible_calls(
    tree: ast.Module,
    *,
    gate: Callable[[ast.Call], bool] | None = None,
    namespace=None,
) -> tuple[ast.Module, list[CallSite]]:
    """Return ``(rewritten_copy, sites)``; *tree* is left untouched.

    Each eligible call has its **callee expression** wrapped, and is handed the
    index of its own :class:`CallSite`::

        compute(x)  ->  __cash_call__(compute, 0)(x)

    The argument list is not rewritten at all, so ``*args``/``**kwargs``,
    keyword arguments and evaluation order need no special handling — and,
    critically, the call stays exactly where it was in the expression. A
    short-circuited ``g()`` in ``f() or g()`` is still only reached when
    ``f()`` is falsy; hoisting it into a temporary would have run it
    unconditionally.

    The copy matters: the caller keeps using the original tree for analysis and
    cache keying, and rewriting in place would desync the runtime's source from
    the upstream simulator's. That is what keeps the runtime and the
    simulator computing the same key without the simulator needing to know
    interception exists.

    ``gate``, when given, is consulted for every structurally-eligible call and
    must return ``True`` for the call to actually be wrapped. It is an
    ADDITIONAL filter, not a replacement for the free-variable rule enforced by
    :func:`eligible_call_nodes` — that structural rule still runs first, and a
    site the gate rejects is simply never wrapped, i.e. left calling the
    original callee directly, at no runtime cost.

    A call that is not wrapped -- the gate rejects it, or *namespace* shows its
    callee is one :func:`interceptable` refuses (a builtin or class such as
    ``dict``, a bound method, a ``@stateful`` function) -- is searched INSIDE
    instead. ``rows.append(dict(k=k, err=score(df, k)))`` accepted the
    ``dict(...)`` as the outermost call; at runtime it is a class and was not
    wrapped, and ``score(df, k)`` inside it was never considered -- a
    backtest was recomputed in full on an unchanged re-run.

    A gate with a ``local`` parameter is also handed the names an enclosing
    comprehension or lambda binds around the call: they have no lineage, and
    the key holds their values (``CallSite.local_arg_positions``).
    """

    try:
        gate_takes_local = gate is not None and "local" in inspect.signature(gate).parameters
    except (TypeError, ValueError):
        gate_takes_local = False

    def skip(call: ast.Call, local: frozenset[str] = frozenset()) -> bool:
        if namespace is not None:
            callee = _static_callee(call.func, namespace)
            if callee is not _NOT_FOUND and not interceptable(callee):
                return True
        if gate is None:
            return False
        return not (gate(call, local=local) if gate_takes_local else gate(call))

    new_tree = copy.deepcopy(tree)
    sites: list[CallSite] = []
    seen: Counter[str] = Counter()
    for stmt in new_tree.body:
        in_loop_unit = isinstance(stmt, ast.For) and namespace is not None
        calls = _eligible_calls_in_scope(stmt, skip, namespace=namespace)
        if not calls:
            continue
        # Computed ONCE per enclosing statement, before any call inside it is
        # rewritten below -- so it reflects the statement exactly as it read
        # before this pass touched it, and is identical whether the statement
        # contains one eligible call or several. See `CallSite.stmt_identity`
        # for why `ast.unparse` (comment-free, so the loop's own
        # `# __iteration_context__:` prefix never reaches it) and why a
        # failure here degrades to `""` -- "not folded into the key" -- rather
        # than aborting the whole rewrite.
        try:
            stmt_identity = ast.unparse(stmt)
        except (ValueError, TypeError, AttributeError, RecursionError):  # degrade, never let keying break the call
            stmt_identity = ""
        for call, local in calls:
            source = ast.unparse(call)
            index = seen[source]
            seen[source] += 1
            sites.append(
                CallSite(
                    source=source,
                    free_names=frozenset(names_read(call) - local),
                    occurrence_index=index,
                    computed_arg_positions=_computed_arg_positions(call, local),
                    has_unpacking=_call_has_unpacking(call),
                    stmt_identity=stmt_identity,
                    local_arg_positions=_local_arg_positions(call, local),
                    content_names=_content_names(call, local),
                    name_arg_positions=_name_arg_positions(call, local),
                    content_source=_content_source(call, local),
                    in_loop_unit=in_loop_unit,
                )
            )
            call.func = ast.Call(
                func=ast.Name(id=HELPER_NAME, ctx=ast.Load()),
                args=[call.func, ast.Constant(value=len(sites) - 1)],
                keywords=[],
            )
    if sites:
        ast.fix_missing_locations(new_tree)
    return new_tree, sites


def _call_has_unpacking(call: ast.Call) -> bool:
    """True when *call* uses ``*args``/``**kwargs`` unpacking.

    An ``ast.Starred`` positional (``f(*xs)``) or a keyword with ``arg=None``
    (``f(**kw)``) means the number of arguments actually passed at runtime is
    not knowable from the AST -- ``len(call.args) + len(call.keywords)``
    counts *expressions* in the call, not values. Shared by
    :func:`_computed_arg_positions` (which fails closed to "every position"
    on this case, since it cannot compute reliable positions past an
    unpacking) and :class:`CallSite` construction (whose ``has_unpacking``
    flag tells the runtime half to refuse the site outright rather than trust
    that static, possibly-wrong count).
    """
    return any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords)


def _local_arg_positions(call: ast.Call, local: frozenset[str]) -> tuple[int, ...]:
    """Positions of arguments that read a name bound by an enclosing
    comprehension or lambda (see ``CallSite.local_arg_positions``)."""
    if not local or _call_has_unpacking(call):
        return ()
    values = [*call.args, *(kw.value for kw in call.keywords)]
    return tuple(i for i, v in enumerate(values) if names_read(v) & local)


def _content_names(call: ast.Call, local: frozenset[str]) -> frozenset[str]:
    """Free names *call* reads only inside computed arguments (see
    ``CallSite.content_names``). None under unpacking, whose arguments the
    runtime keys only on what arrived (see ``_content_source``)."""
    elsewhere = names_read(call.func)
    inside: set[str] = set()
    for value in [*call.args, *(kw.value for kw in call.keywords)]:
        if isinstance(value, ast.Name):
            elsewhere.add(value.id)
        else:
            inside |= names_read(value)
    return frozenset(inside - elsewhere - local)


def _content_source(call: ast.Call, local: frozenset[str]) -> str:
    """*call* with its computed arguments as positional placeholders (see
    ``CallSite.content_source``). Bare names stay: one whose value is too big
    to hash keeps its lineage by name, and a placeholder would let
    ``f(A, B)`` and ``f(B, A)`` share a key."""
    if _call_has_unpacking(call):
        # What arrives is only known at run time, and then every value is
        # hashed with its keyword (`CallKeys._build_unpacked_key`): the spelling
        # of the arguments says nothing more.
        try:
            return f"{ast.unparse(call.func)}(*<received>)"
        except (ValueError, TypeError, AttributeError, RecursionError):  # no content key for this site
            return ""
    computed = set(_computed_arg_positions(call, local))
    shape = copy.deepcopy(call)
    for i in range(len(shape.args)):
        if i in computed:
            shape.args[i] = ast.Name(id=f"_arg{i}", ctx=ast.Load())
    offset = len(shape.args)
    for i, kw in enumerate(shape.keywords):
        if offset + i in computed:
            kw.value = ast.Name(id=f"_arg{offset + i}", ctx=ast.Load())
    try:
        return ast.unparse(shape)
    except (ValueError, TypeError, AttributeError, RecursionError):  # degrade to the spelled source
        return ""


def _name_arg_positions(call: ast.Call, local: frozenset[str]) -> tuple[tuple[str, int], ...]:
    """Bare-name arguments *call* reads nowhere else (see
    ``CallSite.name_arg_positions``); the first position of each."""
    if _call_has_unpacking(call):
        return ()
    values = [*call.args, *(kw.value for kw in call.keywords)]
    elsewhere = set(names_read(call.func))
    for value in values:
        if not isinstance(value, ast.Name):
            elsewhere |= names_read(value)
    found: dict[str, int] = {}
    for i, value in enumerate(values):
        if isinstance(value, ast.Name) and value.id not in local and value.id not in elsewhere:
            found.setdefault(value.id, i)
    return tuple(found.items())


def _computed_arg_positions(call: ast.Call, local: frozenset[str] = frozenset()) -> tuple[int, ...]:
    """Positions, in ``(*args, *kwargs.values())`` order, of non-``ast.Name``
    arguments -- the ones whose evaluated value (not a name's lineage) must be
    hashed into the cache key. A bare name bound by an enclosing comprehension
    or lambda (*local*) is one of them: it has no lineage to resolve.

    ``*args``/``**kwargs`` unpacking makes the position of any later argument
    unreliable to compute here, since the unpacked collection's length is not
    known statically. Fail closed: record every position as computed. (The
    runtime half does not actually attempt to hash these -- ``CallSite.
    has_unpacking`` makes it refuse the whole site instead; this fallback
    value only matters if something ever reads ``computed_arg_positions``
    without checking ``has_unpacking`` first.)
    """
    if _call_has_unpacking(call):
        return tuple(range(len(call.args) + len(call.keywords)))
    positions = []
    for i, arg in enumerate(call.args):
        if not isinstance(arg, ast.Name) or arg.id in local:
            positions.append(i)
    offset = len(call.args)
    for i, kw in enumerate(call.keywords):
        if not isinstance(kw.value, ast.Name) or kw.value.id in local:
            positions.append(offset + i)
    return tuple(positions)


#: Statement shapes the free-variable rule is sound for. Everything else --
#: compound statements and definitions -- is declined; see the note in
#: :func:`eligible_call_nodes`.
_SIMPLE_STATEMENTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr)


def eligible_call_nodes(stmt: ast.stmt) -> list[ast.Call]:
    """Return the calls in *stmt* that may be cached independently of it.

    Outermost-first, in source order. Once a call is accepted its CALLEE
    expression is not searched again -- that is the site's own. Its arguments
    are, because they run whether the outer call hits or not, so an expensive
    call nested there is work no outer entry ever saves.

    **Only simple statements are searched, and that is a safety rule rather
    than a simplification.** (A loop run as one unit is searched per body
    statement by ``_eligible_calls_in_loop``, which applies this rule to each.)
    Cash can execute a loop as a single unit, in which
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
    return [call for call, _local in _eligible_calls_in_scope(stmt)]


def _eligible_calls_in_scope(
    stmt: ast.stmt,
    skip: Callable[[ast.Call, frozenset[str]], bool] | None = None,
    namespace: Mapping[str, object] | None = None,
) -> list[tuple[ast.Call, frozenset[str]]]:
    """:func:`eligible_call_nodes`, each call paired with the names an enclosing
    comprehension or lambda binds around it (see ``CallSite.local_arg_positions``).
    A call *skip* returns ``True`` for is not taken; its inside is searched.

    With *namespace*, a ``for`` statement is searched too (see
    :func:`_eligible_calls_in_loop`)."""
    if isinstance(stmt, ast.For) and namespace is not None:
        return _eligible_calls_in_loop(stmt, skip, namespace)
    if not isinstance(stmt, _SIMPLE_STATEMENTS):
        return []
    targets = _target_names(stmt)
    found: list[tuple[ast.Call, frozenset[str]]] = []
    for root in _search_roots(stmt):
        _collect(root, targets, found, frozenset(), skip)
    return found


def _eligible_calls_in_loop(
    loop: ast.For,
    skip: Callable[[ast.Call, frozenset[str]], bool] | None,
    namespace: Mapping[str, object],
) -> list[tuple[ast.Call, frozenset[str]]]:
    """The eligible calls of each simple statement in a loop cached as ONE unit.

    A long loop runs as a single statement (``single_unit_policy.
    should_run_as_single_unit``), and calls inside it never reached
    the interceptor: fixing one store's data in a 360-iteration fit loop
    re-fitted all 720 models. The free-variable rule stays per statement, as
    per-iteration decomposition applies it: each body statement is searched
    against its own targets, and an expression statement's own call is still
    its effect, never taken (``log_it(x)``).

    What the unit adds is scope. A name the loop binds or writes anywhere has
    no lineage of its own inside it -- the unit's lineage is the whole loop's
    -- so it is local, as a comprehension's variable is: an argument reading it
    is keyed on its value, and a callee reading it is not taken. Neither is a
    call whose callee, or a function or class it reaches, reads such a name as
    a global (``def scaled(v): return sum(v) * FACTOR`` with ``FACTOR`` set in
    the loop): nothing in the key could see it. A callee that cannot be found
    statically is not taken either.
    """
    written = frozenset(_loop_bound_names(loop))
    # And every plain name an argument reads. The unit updates no lineage
    # between iterations, so a global some other call in the loop mutates
    # (`update(conf)`, `bump()`) would be keyed on its state before the loop.
    # Its value, hashed per call, cannot be. The runtime keys such a call on
    # content or not at all (`CallSite.in_loop_unit`).
    local = written | frozenset(_loop_argument_names(loop, namespace))
    reached: dict[int, bool] = {}

    def reads_loop_names(call: ast.Call) -> bool:
        callee = _static_callee(call.func, namespace)
        if callee is _NOT_FOUND:
            return True
        key = id(callee)
        if key not in reached:
            try:
                reached[key] = bool(global_names_reached(callee) & written)
            except Exception:  # noqa: BLE001 - unknown reach: not taken
                reached[key] = True
        return reached[key]

    def loop_skip(call: ast.Call, inner: frozenset[str]) -> bool:
        if reads_loop_names(call):
            return True
        return skip is not None and skip(call, inner)

    found: list[tuple[ast.Call, frozenset[str]]] = []
    for stmt in _loop_simple_statements(loop.body):
        targets = _target_names(stmt)
        for root in _search_roots(stmt):
            _collect(root, targets, found, local, loop_skip)
    return found


def _loop_argument_names(loop: ast.For, namespace: Mapping[str, object]) -> set[str]:
    """Names read inside any call's arguments in *loop*, other than modules,
    classes and functions (those are code, keyed as such)."""
    names: set[str] = set()
    for node in ast.walk(loop):
        if not isinstance(node, ast.Call):
            continue
        for value in [*node.args, *(kw.value for kw in node.keywords)]:
            for name in names_read(value):
                bound = namespace.get(name, _NOT_FOUND)
                if isinstance(bound, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)):
                    continue
                names.add(name)
    return names


def _loop_simple_statements(body: list[ast.stmt]):
    """Simple statements of a loop body, through nested ``for`` and ``if``."""
    for stmt in body:
        if isinstance(stmt, _SIMPLE_STATEMENTS):
            yield stmt
        elif isinstance(stmt, (ast.For, ast.If)):
            yield from _loop_simple_statements(stmt.body)
            yield from _loop_simple_statements(stmt.orelse)


def _loop_bound_names(loop: ast.For) -> set[str]:
    """Every name *loop* binds, deletes or writes into, anywhere inside it."""
    names: set[str] = set()
    for node in ast.walk(loop):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.stmt) and isinstance(node, _SIMPLE_STATEMENTS):
            names |= _target_names(node)
    return names


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


_LOCAL_SCOPES = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp, ast.Lambda)


def _bound_names(node: ast.AST) -> set[str]:
    """The names a comprehension's ``for`` targets or a lambda's parameters bind."""
    if isinstance(node, ast.Lambda):
        a = node.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *(p for p in (a.vararg, a.kwarg) if p)]
        return {p.arg for p in params}
    return {n.id for gen in node.generators for n in ast.walk(gen.target) if isinstance(n, ast.Name)}


def _collect(
    node: ast.AST,
    targets: set[str],
    found: list,
    local: frozenset[str],
    skip: Callable[[ast.Call, frozenset[str]], bool] | None = None,
) -> None:
    if isinstance(node, _LOCAL_SCOPES):
        local = local | _bound_names(node)
    if (
        isinstance(node, ast.Call)
        and not (names_read(node) & targets)
        # A callee that reads a comprehension's own variable is a different
        # callable per element (`m.predict(X)` over `models.items()`), and
        # nothing in the key can see which: never intercepted.
        and not (names_read(node.func) & local)
        and not (skip is not None and skip(node, local))
    ):
        found.append((node, local))
        # Accepted -- its callee expression is now this site's, so nothing
        # there may be taken again. Its ARGUMENTS are a different matter:
        # wrapping a call replaces the callee only, so an argument runs
        # whether the outer call hits or not, and an expensive one nested
        # there was never reused.
        for child in node.args + [kw.value for kw in node.keywords]:
            _collect(child, targets, found, local, skip)
        return
    for child in ast.iter_child_nodes(node):
        _collect(child, targets, found, local, skip)


def names_read(node: ast.AST) -> set[str]:
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
