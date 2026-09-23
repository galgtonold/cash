"""Routing a statement's calls through the call cache.

Each eligible call in a statement is rewritten to resolve its callee through
:class:`~cash.notebook.call_interception.CallCache`, so an expensive call is
cached on its own even where the statement around it cannot be. This module
owns that rewrite and what the call units read while the statement runs:
its TTL, its persist request, and the variables of the loops around it.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from cash.analysis.code_analyzer import CodeAnalyzer
from cash.control_markers import strip_markers
from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import HELPER_NAME, CallCache, wrap_eligible_calls
from cash.notebook.call_refs import with_call_refs
from cash.notebook.call_unit import call_cost_floor_s, call_site_is_cacheable
from cash.tracking.file_tracker import tracking_seconds

if TYPE_CHECKING:
    from cash.notebook._protocols import ShellProtocol, TrackingState
    from cash.tracking.function_tracker import FunctionTracker

logger = logging.getLogger(__name__)

_LOG_PROCESSOR = "[PROCESSOR]"

__all__ = ["CallRouting"]


def _plain_call_assignment(code: str) -> tuple[str, dict[str, int] | None] | None:
    """``(call source, {name: position} or None)`` when *code* is nothing but
    ``name = call(...)`` (``None`` positions) or ``a, b = call(...)``; else
    ``None``. Nothing runs after that call returns but binding the names."""
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return None
    if len(body) != 1:
        return None
    node = body[0]
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target = node.target
    else:
        return None
    if not isinstance(node.value, ast.Call):
        return None
    if isinstance(target, ast.Name):
        return ast.unparse(node.value), None
    if isinstance(target, ast.Tuple | ast.List) and all(isinstance(e, ast.Name) for e in target.elts):
        positions: dict[str, int] = {}
        for position, element in enumerate(target.elts):
            positions[element.id] = position  # a name bound twice keeps the last
        return ast.unparse(node.value), positions
    return None


class CallRouting:
    """The call cache, and the per-statement state its call units read."""

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        function_tracker: FunctionTracker | None,
        compute_hash: Callable[[Any], str] | None,
        *,
        cash_instance: Callable[[], Any | None],
        is_stateful_call: Callable[[str], bool],
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.function_tracker = function_tracker
        self.compute_hash = compute_hash
        #: The Cash instance calls are cached with, resolved per statement.
        self._cash_instance = cash_instance
        self._is_stateful_call = is_stateful_call
        # Sub-expression caching built on first use. Interception
        # is the default; ``# @cash:no-cache-calls`` is the escape hatch. Held
        # with the Cash instance it wraps so a ``reset_session()`` that swaps
        # the instance rebuilds it rather than resolving callees against a
        # dead backend.
        self._call_cache: Any | None = None
        self._call_cache_owner: Any | None = None
        # The TTL in force for the statement currently being processed, read
        # at invoke time by the call units inside it. Refreshed on
        # every statement; `None` means no TTL.
        self.ttl: int | None = None
        # Same shape, same lifetime, for `# @cash:persist` / `%cash_persist`.
        # `False` means "leave it to the cost model".
        self.persist: bool = False
        # Stack-shaped: the non-dunder half of the enclosing for-loop's
        # iteration context, pushed/popped by ``for_handler.py`` around each
        # iteration's body statements (see ``ForLoopHandler._process_one_iteration``)
        # and read by an intercepted call's key build
        # (``call_unit.call_cache_key``'s ``loop_vars``) via
        # :meth:`current_loop_vars_for_call_key`. A stack rather than a single
        # slot because loop bodies nest; the stack depth is the lexical nesting
        # depth (see :meth:`_depth_keyed_loop_scope`). Empty outside any loop,
        # which is the correct "no discriminator to add" answer for a bare
        # statement.
        self._loop_vars: list[dict[str, Any]] = []
        # Parallel stack, pushed/popped in lockstep with ``_loop_vars``
        # by the SAME ``loop_vars_scope`` call: ``{name: full_hash}`` for
        # whichever loop-target names ``for_handler.py`` bound THIS push (its
        # own names only -- deliberately NOT pre-merged with an ancestor
        # loop's digests, unlike ``_loop_vars``). Exists so
        # ``call_unit._loop_var_digest`` can look a digest up instead of
        # recomputing ``compute_hash_full`` on every intercepted call --
        # ``for_handler.py`` already computes this exact hash once per
        # iteration for ``variable_lineage``, and this reuses it.
        #
        # NOT sourced from ``variable_lineage`` itself: that dict is flat,
        # keyed only by name, and never popped -- a nested loop reusing an
        # outer loop's target name (``for t in A: for t in B: pass; call(...)``)
        # leaves ``variable_lineage[name]`` holding the INNER loop's last
        # value for the rest of the OUTER iteration, and a call after the
        # inner loop ends reads that stale entry. First-run wrongness, found
        # live via a real-kernel repro. This stack has the scope discipline
        # ``variable_lineage`` lacks -- it is popped when an iteration's body
        # finishes, restoring whatever level was beneath it -- so
        # :meth:`current_loop_var_digests_for_call_key` can never see a name's
        # digest outlive the scope that produced it.
        self._loop_var_digests: list[dict[str, str]] = []
        # Statement code (context markers stripped) whose calls are not worth
        # routing through the call cache -- see :meth:`code_and_tree_for_execution`.
        self._calls_not_worth_wrapping: set[str] = set()
        # Which statement's calls the call cache's "returned last" is about
        # (:meth:`plain_call_result`): set only when that statement's are wrapped.
        self._calls_wrapped_for: str | None = None

    def begin_statement(self, ttl: int | None, persist: bool) -> None:
        """Publish the statement's TTL and persist request for the calls
        inside it. Set on EVERY statement, so a previous statement's value can
        never leak into one that carries no annotation."""
        self.ttl = ttl
        self.persist = persist

    def begin_cell(self) -> None:
        """A cell starts."""
        if self._call_cache is not None:
            self._call_cache.begin_cell()

    @property
    def in_loop(self) -> bool:
        """Whether a loop iteration's body is running."""
        return bool(self._loop_vars)

    def with_call_refs(self, variables: dict[str, Any], code: str, referenced: dict[str, int]) -> dict[str, Any]:
        """*variables* with the call results the call cache holds stored by
        reference (see :func:`~cash.notebook.call_refs.with_call_refs`)."""
        if self._call_cache is None:
            return variables
        trusted, unpacked = self.plain_call_result(code)
        return with_call_refs(
            variables, self._call_cache.held_results(), referenced, trusted=trusted, unpacked=unpacked
        )

    @contextmanager
    def loop_vars_scope(
        self,
        loop_vars: dict[str, Any],
        loop_var_digests: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Push *loop_vars* (and, alongside, *loop_var_digests*) for the
        duration of one loop iteration's body.

        Called by ``ForLoopHandler._process_one_iteration`` around the whole
        body-statement loop for one iteration -- not per statement -- so a
        nested control structure (``if``/``try``) or a nested ``for`` inside
        the body still sees the enclosing iteration's vars via
        :meth:`current_loop_vars_for_call_key` for every statement it
        eventually reaches.
        A nested ``for`` pushes its own (already-merged, per
        ``build_iteration_context``) context on top; popping unwinds back to
        this one, so the stack always matches the current lexical nesting.

        ``finally`` guarantees the pop happens even when a body statement
        raises -- a loop body that errors out must not leave a stale entry on
        either stack for whatever runs next in the same kernel. Both stacks
        are pushed and popped together so they can never desync -- there is
        no way to push one without the other.

        *loop_var_digests* is optional (defaults to ``{}``) so every call
        site that only cares about values -- direct tests, anything that
        predates this parameter -- keeps working unchanged; a missing digest
        for a given name just means the read side has nothing for it, and
        ``call_unit._loop_var_digest`` falls through to computing one fresh.

        That read side is :meth:`_depth_keyed_loop_scope` (via
        :meth:`current_loop_vars_for_call_key` /
        :meth:`current_loop_var_digests_for_call_key`), whose only consumer
        is ``CallCache``.
        """
        self._loop_vars.append(loop_vars)
        self._loop_var_digests.append(loop_var_digests or {})
        try:
            yield
        finally:
            self._loop_vars.pop()
            self._loop_var_digests.pop()

    def current_call_ttl(self) -> int | None:
        """The TTL in force for the statement being processed.

        Handed to `CallCache` as its `ttl_provider` as a BOUND METHOD, not a
        lambda over a snapshot: one `CallCache` serves every statement, and
        each statement publishes its own annotation before executing.
        """
        return self.ttl

    def current_call_persist(self) -> bool:
        """Whether the statement being processed asked for disk persistence.

        The twin of :meth:`current_call_ttl`, and handed over the same way, for
        the same reason.  `persist` forces an entry past the ~0.1s
        persistence floor; it reached the STATEMENT entry only.  In the
        global-writing shape -- the callee writes a global, so the statement is skip-cached and
        the call entry is the ONLY thing cached -- that left the annotation
        acting on nothing, and cheap-ish work re-ran after every restart.
        """
        return self.persist

    def _depth_keyed_loop_scope(self) -> tuple[dict[str, Any], dict[str, str]]:
        """``(values, digests)`` for the call-unit key build, each entry keyed
        by ``"{depth}:{name}"`` rather than bare ``name``.

        Every active loop level must reach the key: inside ``for q in A: for
        q in B: pull(h)`` the inner level shadows the outer ``q``, and two
        outer iterations over the same inner sequence would share entries.
        The stack depth is the call's lexical nesting, the same on every run
        and for any iteration order.

        Each depth contributes the union of its values' and digests' names:
        a value with no digest keeps its slot and is hashed fresh (see
        :func:`call_unit._loop_var_digest`) rather than dropping out of the
        key. ``for_handler`` strips dunder names before either stack is
        pushed.
        """
        values: dict[str, Any] = {}
        digests: dict[str, str] = {}
        for depth, (val_level, dig_level) in enumerate(zip(self._loop_vars, self._loop_var_digests)):
            for name in val_level.keys() | dig_level.keys():
                key = f"{depth}:{name}"
                if name in val_level:
                    values[key] = val_level[name]
                if name in dig_level:
                    digests[key] = dig_level[name]
        return values, digests

    def current_loop_vars_for_call_key(self) -> dict[str, Any]:
        """Depth-and-name-keyed loop-var values for the call-unit key build.

        Used as the ``loop_vars_provider`` wired into ``CallCache``. See
        :meth:`_depth_keyed_loop_scope` for why entries are keyed by depth.
        """
        values, _ = self._depth_keyed_loop_scope()
        return values

    def current_loop_var_digests_for_call_key(self) -> dict[str, str]:
        """Depth-and-name-keyed loop-var digests for the call-unit key build.

        The digest counterpart to :meth:`current_loop_vars_for_call_key` --
        see that method and :meth:`_depth_keyed_loop_scope` for the full
        reasoning.
        """
        _, digests = self._depth_keyed_loop_scope()
        return digests

    def drain_call_unit_events(self) -> list:
        """CallUnit's own call log, merged into ``decorator_calls``.

        ``CallCache.resolve`` routes an intercepted call's caching through
        ``CallUnit`` rather than ``Cash``'s decorator-call log, so
        ``drain_decorator_calls()`` alone no longer sees it. Pulled in
        separately here rather than inside the processor's ``get_cash_instance``'s try
        block above, so a failure in one drain never suppresses the other.
        """
        if self._call_cache is None:
            return []
        try:
            return self._call_cache.drain_call_log()
        except (AttributeError, TypeError, RuntimeError):
            logger.debug("%s Failed to drain call-unit log", _LOG_PROCESSOR)
            return []

    def learn_call_wrapping(self, code: str, wall_time: float, calls: list) -> None:
        """Record whether *code*'s calls are worth the call cache next time."""
        try:
            floor = call_cost_floor_s(self._cash_instance())
            hit = any(isinstance(ev, dict) and ev.get("cache_hit") for ev in calls or ())
            key = strip_markers(code)
            if wall_time < floor and not hit:
                self._calls_not_worth_wrapping.add(key)
            else:
                self._calls_not_worth_wrapping.discard(key)
        except Exception:  # noqa: BLE001 - wrapping stays on, which is always safe
            pass

    def code_and_tree_for_execution(
        self, code: str, tree: ast.Module | None, annotation: Any | None
    ) -> tuple[str, ast.Module | None]:
        """The ``(code, tree)`` to execute, with eligible calls routed via cache.

        Interception is the DEFAULT: each eligible call has its
        callee wrapped so it resolves to a cached counterpart at call time —
        ``compute(x)`` becomes ``__cash_call__(compute, 0)(x)``, where ``0`` is
        the index of this call's :class:`CallSite`. That fixes the two
        shapes statement-level caching structurally cannot: an expensive call
        inside an in-place mutation (skip-cached, so never reused) and one
        inside an accumulator fold (cached, but keyed on the running prefix, so
        a reorder re-runs the tail).

        **Both halves are returned, and both are load-bearing.**
        :class:`CodeRunner` compiles the *tree* only when the statement's
        last node is an ``ast.Expr`` (so the value can be echoed); every other
        shape compiles the *code string*. Handing back a rewritten tree alone
        therefore worked for ``out.append(compute(x))`` and was silently
        discarded for ``s += compute(x)`` — the accumulator fold, which is half
        the point of the feature.

        Returns the inputs unchanged when opted out, when nothing is eligible,
        or on any failure: a caching optimisation must never be the reason a
        statement stops running.
        """
        # Two things switch interception off: ``# @cash:no-cache-calls`` (the
        # targeted escape hatch) and ``# @cash:no-cache`` (which is an
        # instruction about the WHOLE statement — caching the expensive call
        # inside it honours the letter while breaking the intent, so no-cache
        # wins, exactly as it already wins over ``persist``). Absent an
        # annotation at all, neither opt-out is set, so interception proceeds.
        if annotation is not None and (
            getattr(annotation, "no_cache_calls", False) or getattr(annotation, "no_cache", False)
        ):
            return code, tree
        # Which statement's calls the call cache's "returned last" is about
        # (`plain_call_result`): set below only when this one's are wrapped.
        self._calls_wrapped_for = None
        cash_instance = self._cash_instance()
        if cash_instance is None:
            return code, tree
        # A call cannot take longer than the statement it is in, and one under
        # the call cost floor is never stored. So a statement that last ran
        # under that floor with no call HIT inside it has nothing worth routing
        # through the call cache, and rewriting it is pure overhead: a copy of
        # its tree, an unparse and a gate per call, on every loop iteration --
        # 1.7 of a 631-iteration loop's 9.5 s. Learned in
        # `learn_call_wrapping`; a hit or a slow run clears it again.
        if strip_markers(code) in self._calls_not_worth_wrapping:
            return code, tree
        try:
            # The object-level half of the gate: a call that
            # is structurally eligible (its free variables don't read the
            # statement's own target) can still be uncacheable for every reason
            # a statement can be -- a forbidden call, an untracked input, a
            # user-ns shape ``decide_cacheability`` refuses. Judging it by the
            # SAME rules as the statement containing it (rather than not judging
            # it at all) is what ``call_site_is_cacheable`` exists for; wiring it
            # here means an eligible-but-uncacheable site is never wrapped in
            # the first place, so a doomed key is never even attempted.
            def gate(call: ast.Call, local: frozenset[str] = frozenset()) -> bool:
                # `call_site_is_cacheable` runs the full `decide_cacheability`
                # / `analyze_statement` / `scan_for_forbidden_functions` stack
                # against a bare `ast.Expr(Call)` sub-expression -- a shape the
                # analyzer has never been exercised against before this gate
                # existed. The outer `try` below only ever guarded a copy and
                # an unparse, so it only catches (SyntaxError, ValueError,
                # TypeError, AttributeError); anything else escaping THIS
                # function would surface as the user's own traceback on their
                # statement. Fail closed instead: an
                # exception here means "don't wrap", exactly like a `False`
                # verdict, never "crash the cell".
                try:
                    return call_site_is_cacheable(
                        call,
                        user_ns=self.shell.user_ns,
                        annotation=annotation,
                        # The runtime lineage table genuinely exists at this
                        # call site (unlike the AST-only rewrite-time case
                        # `call_site_is_cacheable`'s docstring justifies
                        # omission for) -- passing it lets the missing-lineage
                        # reason source apply here too, tightening the gate to
                        # the same standard the statement itself is judged by.
                        variable_lineage=self.tracking_state.variable_lineage,
                        is_stateful_call=self._is_stateful_call,
                        scan_forbidden=CodeAnalyzer.scan_for_forbidden_functions,
                        local_names=local,
                    )[0]
                except Exception:  # noqa: BLE001 - fail closed to "don't wrap"
                    # Not `ast.unparse(call)` in this message: that can itself
                    # raise, and doing so here would defeat the very fix this
                    # except exists to provide.
                    logger.debug(
                        "%s cache-calls gate raised; leaving a call site unwrapped",
                        _LOG_PROCESSOR,
                    )
                    return False

            rewritten, sites = wrap_eligible_calls(
                tree if tree is not None else ast.parse(code),
                gate=gate,
                namespace=self.shell.user_ns,
            )
            if not sites:
                # Under default-on, "nothing here was eligible" is the
                # ordinary case (most statements have no expensive sub-call),
                # not a mistake worth a warning.
                return code, tree
            new_code = ast.unparse(rewritten)
            # ``ast.unparse`` drops a trailing ";", which CodeRunner
            # reads as "suppress the repr". Losing it would make a rewritten
            # statement echo a value the user silenced.
            if code.rstrip().endswith(";"):
                new_code += ";"
            if self._call_cache is None or self._call_cache_owner is not cash_instance:
                self._call_cache = CallCache(
                    cash_instance,
                    ttl_provider=self.current_call_ttl,
                    persist_provider=self.current_call_persist,
                    ctx_provider=lambda: CacheKeyContext(
                        variable_lineage=self.tracking_state.variable_lineage,
                        user_ns=self.shell.user_ns,
                        function_tracker=self.function_tracker,
                        compute_hash_fn=self.compute_hash,
                    ),
                    # `self.current_loop_vars_for_call_key` (bound method, not
                    # a lambda capturing a snapshot) so it re-reads
                    # `_loop_vars` at INVOKE time -- the `CallCache`
                    # instance is reused across statement executions (guarded
                    # by `_call_cache_owner` above), but the loop this call
                    # sits in pushes/pops its vars fresh on every iteration.
                    #
                    # Depth-and-name-keyed: a call INSIDE a
                    # loop that reuses an ancestor's target name needs BOTH
                    # scopes' entries to survive at once -- see
                    # `_depth_keyed_loop_scope`'s docstring.
                    loop_vars_provider=self.current_loop_vars_for_call_key,
                    loop_var_digests_provider=self.current_loop_var_digests_for_call_key,
                )
                self._call_cache_owner = cash_instance
            plain = _plain_call_assignment(code)
            self._call_cache.set_sites(sites, plain_value_source=plain[0] if plain else None)
            self._calls_wrapped_for = code
            self.shell.user_ns[HELPER_NAME] = self._call_cache.resolve
            return new_code, rewritten
        except (SyntaxError, ValueError, TypeError, AttributeError):
            logger.debug("%s cache-calls rewrite failed; executing unmodified", _LOG_PROCESSOR)
            return code, tree

    def cash_time_marks(self) -> tuple[float, Any, float, float]:
        """Cash's own clocks, read around a statement (see :meth:`statement_cost`)."""

        unit = self._call_cache.call_unit if self._call_cache is not None else None
        return (tracking_seconds(), unit, getattr(unit, "overhead_s", 0.0), getattr(unit, "hits_saved_s", 0.0))

    def _statement_tax(self, marks: tuple[float, Any, float, float]) -> tuple[float, float]:
        """``(cash's own seconds inside this statement, what its cached calls saved)``.

        The tax is time recording file reads, and keying, hashing and storing
        the calls cash routed -- work the user's own kernel would not have
        done. It is measured, not estimated: the file tracker and the call
        unit both count their own seconds.

        Used twice, and the two must not diverge: to price the statement for
        storing (:meth:`statement_cost`) and to report it as OVERHEAD rather
        than as the user's compute. Counting it as compute cancelled it out of
        `%cash_stats`, which reported 210 s of overhead for a run a pairing
        measured 370 s slower.
        """

        tracking0, unit0, overhead0, saved0 = marks
        tracking = max(0.0, tracking_seconds() - tracking0)
        unit = self._call_cache.call_unit if self._call_cache is not None else None
        overhead = saved = 0.0
        if unit is not None:
            base_overhead, base_saved = (overhead0, saved0) if unit is unit0 else (0.0, 0.0)
            overhead = max(0.0, getattr(unit, "overhead_s", 0.0) - base_overhead)
            saved = max(0.0, getattr(unit, "hits_saved_s", 0.0) - base_saved)
        return tracking + overhead, saved

    def statement_cost(self, wall_time: float, marks: tuple[float, Any, float, float]) -> float:
        """What the statement's own code cost, for storing and for crediting a hit.

        The wall time under cash, less cash's own time inside it -- recording
        file reads, keying and storing the calls it routed -- plus what the calls
        it served from the cache would have cost. The badge once said "saved 16.50s" for a
        folder read that takes 1.8 s without cash, and "saved 6.55s" for
        a dict of fits that takes 50-100 s, built from calls served from the
        cache. The badge's run time stays the wall time.
        """
        try:
            tax, saved = self._statement_tax(marks)
            return max(0.0, wall_time - tax) + saved
        except Exception:  # noqa: BLE001 - a cost estimate never breaks a statement
            return wall_time

    def cash_tax_seconds(self, marks: tuple[float, Any, float, float]) -> float:
        """The tax alone, for the session's overhead accounting."""
        try:
            return self._statement_tax(marks)[0]
        except Exception:  # noqa: BLE001 - never let accounting break a statement
            return 0.0

    def plain_call_result(self, code: str) -> tuple[tuple[str, int] | None, dict[str, int] | None]:
        """``(trusted, unpacked)`` for `with_call_refs` when the statement is
        ``name = call(...)`` or ``a, b = call(...)``: nothing but binding the
        names runs after that call returns, so its result is known unchanged
        without digesting it again. ``(None, None)`` otherwise."""
        if self._calls_wrapped_for != code:
            return None, None
        outermost = getattr(self._call_cache, "outermost_result", None)
        found = outermost() if callable(outermost) else None
        if not found:
            return None, None
        plain = _plain_call_assignment(code)
        # The call that returned last must be the statement's value itself:
        # in ``x = f(g(y))`` with ``f`` not wrapped, ``g`` returned last, and
        # ``f`` may have changed that result and handed it back.
        if plain is None or plain[0] != found[2]:
            return None, None
        return (found[0], found[1]), plain[1]
