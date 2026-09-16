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

**Caveat for large loops.** The ``out.append(compute(x))`` example above only
reaches this module when the loop is decomposed per-iteration.
``for_handler._should_execute_loop_as_single_unit`` routes a large-enough loop
to the single-unit fast path instead, gated by
``_MIN_ITERATIONS_FOR_SINGLE_UNIT``, ``_PER_STMT_OVERHEAD_SEC``, and
``_MIN_OVERHEAD_SEC`` in that module. Calls inside a single-unit loop never
reach the interceptor at all. For the precise threshold, see
``docs/known-limitations.md``'s "A long for-append loop can stop caching"
section, whose numbers are pinned to those constants by a claim-anchor test
(``tests/docs/test_claim_anchors.py``).
"""

import ast
import copy
import types
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cache_key import CacheKeyContext

__all__ = [
    "eligible_call_nodes", "wrap_eligible_calls", "CallCache", "CallSite", "HELPER_NAME",
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
    #: a correctness requirement rather than a tuning knob. See Task 3.
    computed_arg_positions: tuple[int, ...] = ()
    #: True when the call uses ``*args``/``**kwargs`` unpacking. The runtime
    #: half (``CallUnit``) must refuse to key such a call at all: the live
    #: arity it sees (the flattened ``(*args, *kwargs.values())`` it is
    #: actually called with) can differ from ``len(computed_arg_positions)``,
    #: which is a STATIC count from the AST -- hashing only the positions that
    #: count predicts would silently ignore any unpacked elements beyond it.
    #: See CAS-243 review C2.
    has_unpacking: bool = False
    #: ``ast.unparse`` of the statement that CONTAINS this call -- computed
    #: once per enclosing statement, before any call inside it is rewritten
    #: (CAS-256). ``call_cache_key``'s base key is built from the call's OWN
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
    #: changes for every iteration on a reorder (CAS-242). If that raw text
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
    #: first's result -- on the first run with a global ``v`` around (round 22,
    #: a grid search per model handed the SVM the logistic regression). These
    #: are also in ``computed_arg_positions``, and are hashed in full, never
    #: sampled, for the reason a loop variable is (see
    #: ``call_unit._loop_var_digest``): the argument's value is then all that
    #: tells the elements apart. That holds for an argument computed from the
    #: name too -- ``fit_score(make_features(cleaned[mid], W))`` (r23s3): a
    #: frame's sample is its shape, dtypes and first five rows, and rolling
    #: features all begin with the same empty rows.
    local_arg_positions: tuple[int, ...] = ()
    #: Free names the call reads only inside computed arguments -- ``cleaned``
    #: and ``make_features`` in ``fit_score(make_features(cleaned[mid], W))``.
    #: What they contribute is the argument's value, which the key can hash
    #: instead of their lineage (see ``call_unit.call_cache_key``'s
    #: *by_content*): a change to ``cleaned`` that leaves this element's
    #: features as they were then keeps the call's result.
    content_names: frozenset[str] = frozenset()
    #: ``(name, position)`` of each argument passed as a bare name that the
    #: call reads nowhere else -- ``PARAMS`` and ``cutoff`` in
    #: ``fit_series(g, PARAMS, cutoff)``. Under content keying the key may
    #: hold such a value instead of the name's lineage (see
    #: ``call_unit.call_cache_key``'s *name_digests*).
    name_arg_positions: tuple[tuple[str, int], ...] = ()
    #: The call as its content key sees it: each computed argument replaced by
    #: a placeholder for its position (``fit_score(_arg0)``). Under content
    #: keying that argument's value is in the key, so how it was spelled is
    #: not: the sweep's ``make_features(cleaned[mid], W)`` and the pick's
    #: ``make_features(cleaned[mid], BEST_W)`` hand ``fit_score`` the same
    #: features, and keyed on the text the pick re-fitted all of them
    #: (round 25, r25s3). Empty under unpacking, which is never content-keyed.
    content_source: str = ""


def interceptable(fn) -> bool:
    """Whether a callee is one :meth:`CallCache.resolve` wraps.

    Only a plain Python function: builtins, classes and bound methods are left
    alone (see "Bound methods are deliberately not intercepted" in the docs).
    Also not one that is already ``@cash.cache``-d, nor ``@cash.stateful`` --
    THE documented way to say "never cache this"; skipping that check cached a
    stateful callee and returned a stale value on the FIRST run (``[1, 1]``
    where plain Python gives ``[1, 2]``), because the second call in a loop hit
    the entry the first had just written. Nor cash's own instrumentation:
    ``file_tracker`` replaces ``open``, ``pd.read_csv`` and friends with
    tracking wrappers, which are plain functions; wrapping one means trying to
    cache a file handle (CAS-246). The sentinel is the one ``cache_key.py``
    already reads (CAS-214), and every install site sets it.
    """
    return (isinstance(fn, types.FunctionType)
            and not getattr(fn, '_cash_cached', False)
            and not getattr(fn, '_cash_stateful', False)
            and not getattr(fn, '_is_file_tracker_patch', False))


_NOT_FOUND = object()


def _static_callee(node: ast.AST, namespace) -> object:
    """The object a callee expression names, found without calling anything.

    A bare name through *namespace* and the builtins; an attribute only through
    a module or a class (a static lookup on a class, so no descriptor or
    property runs). Anything else -- an instance's method, a subscript, a call
    result -- is ``_NOT_FOUND``.
    """
    import builtins
    import inspect
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


def _is_storable(result) -> bool:
    """``cache_if`` predicate: may this call's result be written to the cache?

    Refuses objects that are identity-coupled to a library global — today, a
    matplotlib Figure/Axes. The RAM tier deep-copies on store and
    ``Figure.__setstate__`` re-registers the COPY as pyplot's *current figure*,
    so a later bare ``plt.savefig()`` writes the cache's snapshot instead of the
    figure the user drew on, on the FIRST run and silently.

    ``statement/processor.py`` already refuses exactly this shape. Routing calls
    through the decorator skipped that guard, because the decorator never sees a
    statement — adversarial probing produced two different PNGs from one figure
    and ``plt.gcf() is fig`` returning False. Applied as ``cache_if`` rather than
    a post-check so the refusal lands *before* the write, which is what stops
    the deep copy from being made at all.

    Deliberately narrow: only the identity-coupled family. Everything else the
    decorator would cache is still cached.
    """
    try:
        from .cacheability_decision import identity_coupled_reason
        return identity_coupled_reason("<intercepted call>", result) is None
    except Exception:  # noqa: BLE001 - never let the predicate break the call
        return True


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

    def __init__(
        self,
        cash_instance,
        ctx_provider: Callable[[], CacheKeyContext] | None = None,
        loop_vars_provider: Callable[[], dict[str, Any]] | None = None,
        loop_var_digests_provider: Callable[[], dict[str, str]] | None = None,
        ttl_provider: Callable[[], int | None] | None = None,
        persist_provider: Callable[[], bool] | None = None,
    ):
        self._cash = cash_instance
        # Keyed by (id(fn), site) -- NOT (id(fn), site_index). `set_sites` is
        # called once per STATEMENT, so `site_index` (an index into that
        # statement's own site list) is reused across every statement that
        # rewrites at least one call: index 0 means something different on
        # every statement. Keying on the index let editing a cell reuse the
        # PREVIOUS statement's wrapper -- same fn, index 0 -- serving its
        # source/free_names/computed_arg_positions after the callee's own
        # argument expression had changed (CAS-243 review C1; reproduced as
        # `out.append(compute(a + 100))` silently returning `out.append(compute(a))`'s
        # cached value). `CallSite` is a frozen, hashable dataclass of
        # `(source, free_names, occurrence_index, computed_arg_positions,
        # has_unpacking, stmt_identity)`, so keying on the site itself
        # self-invalidates on any of those changing -- while an UNCHANGED cell
        # re-executing still gets a wrapper hit, because `wrap_eligible_calls`
        # builds a NEW CallSite object each time but an EQUAL one (frozen
        # dataclasses hash and compare by value), so the wrapper-reuse
        # optimisation (`test_wrapper_is_reused_for_the_same_function`) is
        # preserved rather than lost to a blanket `_wrappers.clear()` in
        # `set_sites`. A function's id can also be reused after garbage
        # collection, so the original is pinned alongside the wrapper in the
        # value tuple to keep it alive and detect a recycled id.
        #
        # `stmt_identity` (CAS-256) joining this tuple is deliberate, not
        # incidental: two statements that previously built an EQUAL CallSite
        # (same call text, same free names, same occurrence index) now build
        # DIFFERENT ones, so each statement gets its own wrapper instead of
        # silently sharing one minted for the other. That re-scoping is
        # exactly what fixes the underlying key collision -- a shared wrapper
        # closes over one `site`, and a wrapper reused across statements would
        # still build the collapsed key `stmt_identity` exists to prevent.
        self._wrappers: dict[tuple[int, CallSite | None], tuple[types.FunctionType, object]] = {}
        # NOTE: there is deliberately no name-reconciliation here any more.
        # This class used to rebuild ``module.qualname`` via
        # ``Cash._get_func_key`` so the badge could tell an intercepted call
        # from a hand-decorated one, with a comment warning that the two "must
        # agree exactly or the badge silently stops marking intercepted calls".
        # Call-unit events set ``intercepted=True`` at the source, so the two
        # can no longer drift.
        #: The current cell's rewrite-time site table, set by the processor
        #: right before execution via :meth:`set_sites`.
        self._sites: list[CallSite] = []
        # Local import: call_unit.py imports CallSite/_names_read from this
        # module, so a module-level import here would be a circular import at
        # load time. Deferred to first construction instead.
        from .call_unit import CallUnit
        self._call_unit = CallUnit(
            cash_instance,
            ctx_provider or self._default_ctx,
            loop_vars_provider or self._default_loop_vars,
            loop_var_digests_provider or self._default_loop_var_digests,
            # No fallback: absent a live processor there is no annotation in
            # force, and `None` is precisely "no TTL" (CAS-268).
            ttl_provider,
            # Same reasoning for `persist`: no processor means no annotation,
            # and `None` degrades to "don't force it" (CAS-269).
            persist_provider,
        )

    def _default_ctx(self) -> CacheKeyContext:
        """Fallback used only when no live processor state was wired in.

        The production call site (``statement/processor.py``) always supplies a
        real ``ctx_provider`` bound to the executing cell's ``user_ns`` and
        ``variable_lineage``. This empty context is exercised only by
        ``resolve()`` calls that never registered a site (see below) -- direct,
        non-production use of ``CallCache`` -- where it is harmless: lineage
        resolution degrades to id-based hashing rather than a dict lookup, and
        nothing is served incorrectly.
        """
        return CacheKeyContext(variable_lineage={}, user_ns={})

    @staticmethod
    def _default_loop_vars() -> dict[str, Any]:
        """Fallback used only when no live processor state was wired in.

        Same reasoning as :meth:`_default_ctx`: the production call site
        always supplies a real ``loop_vars_provider`` bound to the executing
        statement processor's loop-var stack. ``{}`` here is what
        ``call_cache_key`` already treats as "outside a loop" -- correct,
        merely undiscriminated.
        """
        return {}

    @staticmethod
    def _default_loop_var_digests() -> dict[str, str]:
        """Fallback used only when no live processor state was wired in.

        Same reasoning as :meth:`_default_loop_vars`. ``{}`` here is what
        ``call_cache_key``'s ``_loop_var_digest`` already treats as "no
        precomputed digest" -- it falls through to a fresh
        ``compute_hash_full`` of the value, correct, merely undiscounted.
        """
        return {}

    def set_sites(self, sites: list[CallSite]) -> None:
        self._sites = sites
        # One call per statement run: each site's guard starts over.
        self._call_unit.begin_statement()

    def drain_call_log(self) -> list[dict]:
        """Events :class:`~cash.notebook.call_unit.CallUnit` recorded since the
        last drain, in the same shape ``Cash.drain_decorator_calls`` returns.

        An intercepted call routed through :meth:`resolve`'s real-site branch
        no longer calls ``self._cash.cache`` at all, so nothing about it lands
        in the ``Cash`` instance's own decorator-call log any more -- the
        processor must pull this in and merge it with
        ``drain_decorator_calls()`` or the badge, the ``@cache`` row and
        ``%cash_stats`` silently stop seeing intercepted calls.
        """
        return self._call_unit.drain()

    def resolve(self, fn, site_index: int = 0):
        """Return *fn* or a cached counterpart. Never raises."""
        if not interceptable(fn):
            return fn

        try:
            site = self._sites[site_index]
        except (IndexError, TypeError):
            site = None

        # Keyed on the SITE, not the index -- see the long comment on
        # `_wrappers` in `__init__` for why the index alone is unsafe.
        cache_key = (id(fn), site)
        entry = self._wrappers.get(cache_key)
        if entry is not None and entry[0] is fn:
            return entry[1]

        try:
            if site is not None:
                # The real path (CAS-243 Task 5): key and store through the
                # statement backend via the call's own CallSite.
                wrapper = self._call_unit.wrap(fn, site)
            else:
                # No site registered for this index -- CallCache is being used
                # outside the ``_code_and_tree_for_execution`` rewrite pipeline
                # (e.g. called directly, as every pre-Task-5 unit test does).
                # In production ``set_sites`` is always called with a non-empty
                # list before ``__cash_call__`` is ever bound into ``user_ns``
                # (``_code_and_tree_for_execution`` returns early when
                # ``wrap_eligible_calls`` finds nothing), so this branch is not
                # reachable from real notebook execution. Keep the previously-
                # shipped decorator-based wrapping here rather than passing the
                # callee through unwrapped: an unrecognised shape must degrade
                # to a slower-but-correct cache, not to silently losing caching.
                wrapper = self._cash.cache(fn, cache_if=_is_storable)
        except Exception:  # noqa: BLE001 - a caching wrapper is never worth an error
            return fn
        self._wrappers[cache_key] = (fn, wrapper)
        return wrapper


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
    the upstream simulator's. That is what keeps ADR-007 satisfied without the
    simulator needing to know interception exists.

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
    backtest was recomputed in full on an unchanged re-run (round 22).

    A gate with a ``local`` parameter is also handed the names an enclosing
    comprehension or lambda binds around the call: they have no lineage, and
    the key holds their values (``CallSite.local_arg_positions``).
    """
    import inspect
    try:
        gate_takes_local = gate is not None and 'local' in inspect.signature(gate).parameters
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
        calls = _eligible_calls_in_scope(stmt, skip)
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
        except Exception:  # noqa: BLE001 - degrade, never let keying break the call
            stmt_identity = ""
        for call, local in calls:
            source = ast.unparse(call)
            index = seen[source]
            seen[source] += 1
            sites.append(
                CallSite(
                    source=source,
                    free_names=frozenset(_names_read(call) - local),
                    occurrence_index=index,
                    computed_arg_positions=_computed_arg_positions(call, local),
                    has_unpacking=_call_has_unpacking(call),
                    stmt_identity=stmt_identity,
                    local_arg_positions=_local_arg_positions(call, local),
                    content_names=_content_names(call, local),
                    name_arg_positions=_name_arg_positions(call, local),
                    content_source=_content_source(call, local),
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
    that static, possibly-wrong count -- see CAS-243 review C2).
    """
    return any(isinstance(a, ast.Starred) for a in call.args) or any(
        kw.arg is None for kw in call.keywords
    )


def _local_arg_positions(call: ast.Call, local: frozenset[str]) -> tuple[int, ...]:
    """Positions of arguments that read a name bound by an enclosing
    comprehension or lambda (see ``CallSite.local_arg_positions``)."""
    if not local or _call_has_unpacking(call):
        return ()
    values = [*call.args, *(kw.value for kw in call.keywords)]
    return tuple(i for i, v in enumerate(values) if _names_read(v) & local)


def _content_names(call: ast.Call, local: frozenset[str]) -> frozenset[str]:
    """Free names *call* reads only inside computed arguments (see
    ``CallSite.content_names``). None under unpacking, whose arguments the
    runtime refuses to key at all."""
    if _call_has_unpacking(call):
        return frozenset()
    elsewhere = _names_read(call.func)
    inside: set[str] = set()
    for value in [*call.args, *(kw.value for kw in call.keywords)]:
        if isinstance(value, ast.Name):
            elsewhere.add(value.id)
        else:
            inside |= _names_read(value)
    return frozenset(inside - elsewhere - local)


def _content_source(call: ast.Call, local: frozenset[str]) -> str:
    """*call* with its computed arguments as positional placeholders (see
    ``CallSite.content_source``). Bare names stay: one whose value is too big
    to hash keeps its lineage by name, and a placeholder would let
    ``f(A, B)`` and ``f(B, A)`` share a key."""
    if _call_has_unpacking(call):
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
    except Exception:  # noqa: BLE001 - degrade to the spelled source
        return ""


def _name_arg_positions(call: ast.Call, local: frozenset[str]) -> tuple[tuple[str, int], ...]:
    """Bare-name arguments *call* reads nowhere else (see
    ``CallSite.name_arg_positions``); the first position of each."""
    if _call_has_unpacking(call):
        return ()
    values = [*call.args, *(kw.value for kw in call.keywords)]
    elsewhere = set(_names_read(call.func))
    for value in values:
        if not isinstance(value, ast.Name):
            elsewhere |= _names_read(value)
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
    return [call for call, _local in _eligible_calls_in_scope(stmt)]


def _eligible_calls_in_scope(
    stmt: ast.stmt, skip: Callable[[ast.Call, frozenset[str]], bool] | None = None,
) -> list[tuple[ast.Call, frozenset[str]]]:
    """:func:`eligible_call_nodes`, each call paired with the names an enclosing
    comprehension or lambda binds around it (see ``CallSite.local_arg_positions``).
    A call *skip* returns ``True`` for is not taken; its inside is searched."""
    if not isinstance(stmt, _SIMPLE_STATEMENTS):
        return []
    targets = _target_names(stmt)
    found: list[tuple[ast.Call, frozenset[str]]] = []
    for root in _search_roots(stmt):
        _collect(root, targets, found, frozenset(), skip)
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


_LOCAL_SCOPES = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp, ast.Lambda)


def _bound_names(node: ast.AST) -> set[str]:
    """The names a comprehension's ``for`` targets or a lambda's parameters bind."""
    if isinstance(node, ast.Lambda):
        a = node.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *(p for p in (a.vararg, a.kwarg) if p)]
        return {p.arg for p in params}
    return {n.id for gen in node.generators for n in ast.walk(gen.target) if isinstance(n, ast.Name)}


def _collect(node: ast.AST, targets: set[str], found: list, local: frozenset[str],
             skip: Callable[[ast.Call, frozenset[str]], bool] | None = None) -> None:
    if isinstance(node, _LOCAL_SCOPES):
        local = local | _bound_names(node)
    if (isinstance(node, ast.Call) and not (_names_read(node) & targets)
            # A callee that reads a comprehension's own variable is a different
            # callable per element (`m.predict(X)` over `models.items()`), and
            # nothing in the key can see which: never intercepted.
            and not (_names_read(node.func) & local)
            and not (skip is not None and skip(node, local))):
        found.append((node, local))
        return  # accepted -- do not search inside it
    for child in ast.iter_child_nodes(node):
        _collect(child, targets, found, local, skip)


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
