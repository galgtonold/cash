"""Runtime half of sub-statement caching: keying, gating and storing a call.

``call_interception.py`` owns the AST half — which call nodes are structurally
eligible, and the rewrite. This module owns everything that needs a live
object: the key (which reads variable lineage), the runtime gate, the
post-execution refusals, and the backend round-trip.

**Why this is not the decorator.** ``@cash.cache`` keys a call by pickling
every argument. That is the right contract for a function called from
anywhere, and the wrong one here: inside a notebook the arguments are usually
tracked variables whose lineage is already computed, so the statement path
resolves them with a dict lookup where the decorator would re-hash a whole
DataFrame on every iteration. Routing through ``compute_cache_key`` also means
an intercepted call is judged by the same rules as the statement containing
it, rather than by a stricter analysis that fires on one spelling and not the
other.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import logging
import pathlib as _pathlib
import sys
import time as _time
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from cash._clock import perf_counter as _perf_counter
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability
from cash.notebook._trace import trace_event
from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_effects import (
    UNWRAP_FAILED,
    call_capturing_output,
    capture_globals,
    hash_args,
    replay_deps,
    replay_output,
    restore_globals,
    unwrap_callee_globals,
)
from cash.notebook.call_entries import CallEntries
from cash.notebook.call_interception import CallSite, names_read
from cash.notebook.call_key import CallKeys, callee_mutated_globals, global_digests
from cash.notebook.call_refs import (
    DIGEST_FIELD,
    SIZE_FIELD,
)
from cash.tracking.file_tracker import FileAccessTracker
from cash.tracking.randomness import capture_rng_state, rng_modules_changed
from cash.tracking.tracker_context import active_tracker

logger = logging.getLogger(__name__)

__all__ = ["call_site_is_cacheable", "CallUnit"]


def call_site_is_cacheable(
    call_node: ast.Call,
    *,
    user_ns: Mapping[str, Any],
    annotation: CacheAnnotation | None,
    is_stateful_call: Callable[[str], bool],
    scan_forbidden: Callable[[str, Mapping[str, Any], ast.Module | None], list[str]],
    variable_lineage: Mapping[str, str] | None = None,
    local_names: frozenset[str] = frozenset(),
) -> tuple[bool, list[str]]:
    """Judge one call node by the statement path's own rules.

    The call is wrapped as a one-statement module so :func:`analyze_statement`
    and *scan_forbidden* see exactly what they see for a statement, and the
    decision is delegated to :func:`decide_cacheability` -- the same function
    the statement path calls. A sub-unit is judged by the rules that govern
    the statement containing it, not by a stricter or looser analysis that
    happens to fire on one spelling and not the other.

    ``outputs=set()`` is the one deliberate adaptation. ``decide_cacheability``
    computes ``pure_mutations = top_level_mutated_vars - outputs``; a call
    binds nothing, so an empty *outputs* means ANY detected mutation refuses.
    That is the fail-closed direction and must not be "fixed" by inventing
    outputs for a call site.

    ``variable_lineage`` gates whether the *missing-lineage* reason source is
    even asked. Passing it applies that source using the real inputs read by
    the call. Omitting it (the AST-only half decided at rewrite time, before
    any runtime lineage table exists) asks ``decide_cacheability`` with an
    empty ``inputs`` set instead of a fabricated lineage table -- missing
    lineage is never a reachable reason in that mode. This is not "skipping a
    check cash normally makes": a rewrite-time call site has no reads to
    check against yet, and lineage is asked again, for real, wherever the
    runtime half of this feature evaluates the call.

    ``local_names`` -- what an enclosing comprehension or lambda binds -- are
    not inputs with a lineage to look up: the key holds their values
    (``CallSite.local_arg_positions``). Counted as inputs, they found no
    lineage and refused the call, so a call in a comprehension was cached
    only when a global of the same name happened to exist.
    """
    tree = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
    code = ast.unparse(call_node)
    inputs = names_read(call_node) - local_names if variable_lineage is not None else set()
    return decide_cacheability(
        code=code,
        tree=tree,
        inputs=inputs,
        outputs=set(),
        annotation=annotation,
        analysis=analyze_statement(code, tree, user_ns),
        user_ns=user_ns,
        variable_lineage=variable_lineage if variable_lineage is not None else {},
        is_stateful_call=is_stateful_call,
        scan_forbidden=scan_forbidden,
    )


#: Below this, a call is not worth a key, a store, or a timer.
#:
#: Not the statement path's ``min_execution_time_to_cache_seconds`` (0.01s):
#: the two paths do not have the same overhead, and that floor is ~3x too
#: conservative here -- a whole band of loops would clear neither it nor the
#: single-unit threshold and so cache nothing at all.
#:
#: 3ms is derived from measurement, not from the statement path. End-to-end,
#: n=124 (the scripts in ``benchmarks/call_unit_cost/``): store ~0.7ms/call, hit ~1.2ms/call, so a
#: call pays for itself once its body clears ~1.2ms. Measured warm rerun vs
#: cash-off at this n: 0.1ms body 8x SLOWER, 1ms 1.25x slower, 2ms 1.3x faster,
#: 5ms 5x faster. 3ms keeps ~2.5x margin over the break-even for slower
#: machines while covering the reported band.
#:
#: Unchanged by the lower value: an allow-list of "trivial builtins" is still
#: unnecessary, since ``len()`` cannot clear 3ms either.
#:
#: Bodies BELOW this floor are not left uncovered -- they are the promotion
#: case, where one whole-loop unit amortises over every iteration instead of
#: N per-call entries. This constant is the boundary between
#: the two mechanisms, which is why no band should fall between them.
_COST_FLOOR_S = 0.003


def call_cost_floor_s(cash: Any) -> float:
    """The bar a call's own execution must clear to be stored.

    Read from *cash*'s config on every decision rather than captured, so
    ``cash.configure(call_cost_floor_seconds=...)`` takes effect
    immediately -- the contract ``min_execution_time_to_cache_seconds``
    already has.

    Checked with isinstance rather than ``float()`` in a try/except: a
    MagicMock's ``__float__`` returns 1.0 instead of raising, and
    ``cash_instance`` is a MagicMock throughout the unit suite, so the
    exception form would silently install a 1-SECOND floor there and cache
    nothing. ``_COST_FLOOR_S`` stays the default and the fallback.
    """
    value = getattr(getattr(cash, "config", None), "call_cost_floor_seconds", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return _COST_FLOOR_S


#: The many-cheap-calls guard (``CallUnit._entry_for``), in the numbers
#: ``for_handler`` uses to run a loop as one unit: past 50 calls in one
#: statement run, calls cheaper than this are timed plain on a few samples, and
#: the site runs plain for the rest of the run when caching a call costs more
#: than ``_OVERHEAD_FACTOR`` times what the call computes.
_GUARD_AFTER_CALLS = 50
_GUARD_CHEAP_BELOW_S = 0.05
_PLAIN_SAMPLES = 5
_OVERHEAD_FACTOR = 3.0
#: ... and when keying and looking a call up costs more than this share of the
#: call itself: the hit that follows also restores the value, so it would save
#: next to nothing.
_HIT_MUST_SAVE = 0.75


@dataclasses.dataclass
class _SiteRun:
    """One call site's calls in the statement run under way."""

    calls: int = 0
    total_s: float = 0.0
    #: Of ``total_s``, what building the key and looking it up cost: all a
    #: hit pays, so a call cheaper than it is never worth caching.
    key_s: float = 0.0
    computed: int = 0
    compute_s: float = 0.0
    probing: bool = False
    plain_n: int = 0
    plain_s: float = 0.0
    decided: bool = False
    plain: bool = False


#: Where warnings re-emitted for a cash frame are de-duplicated, per file.
_WARNING_REGISTRIES: dict[str, dict] = {}
_CASH_DIR = _pathlib.Path(__file__).resolve().parents[1]


def _in_cash(filename: str) -> bool:
    try:
        return _pathlib.Path(filename).resolve().is_relative_to(_CASH_DIR)
    except (OSError, ValueError, TypeError):
        return False


@contextlib.contextmanager
def _warnings_at_the_caller():
    """Re-emit warnings raised inside an intercepted call, at the user's line.

    ``warnings.warn(..., stacklevel=2)`` names the frame that called the
    function -- which, under interception, is cash's wrapper. A pandas warning
    quoted ``result = fn(*args, **kwargs)`` from ``call_unit.py`` where the
    user's own line belonged. Recorded, then re-emitted with
    a cash frame replaced by the first user frame above it, de-duplicated per
    location as the default filter would. A filter that turns warnings into
    errors is left to act as it would: recording would swallow the exception.
    """
    if any(
        action == "error" and category is Warning and message is None and module is None
        for action, message, category, module, _lineno in warnings.filters
    ):
        yield
        return
    catcher = warnings.catch_warnings(record=True)
    caught = catcher.__enter__()
    warnings.simplefilter("always")
    try:
        yield
    finally:
        catcher.__exit__(None, None, None)
        for w in caught:
            filename, lineno = w.filename, w.lineno
            if _in_cash(filename):
                frame = sys._getframe(1)
                while frame is not None and (
                    _in_cash(frame.f_code.co_filename) or frame.f_code.co_filename == contextlib.__file__
                ):
                    frame = frame.f_back
                if frame is not None:
                    # The line is read from linecache, where cash registered
                    # the statement; passing the frame's globals asks for a
                    # module loader a cell does not have.
                    filename, lineno = frame.f_code.co_filename, frame.f_lineno
            try:
                warnings.warn_explicit(
                    w.message,
                    w.category,
                    filename,
                    lineno,
                    registry=_WARNING_REGISTRIES.setdefault(filename, {}),
                    source=w.source,
                )
            except Exception:  # noqa: BLE001 - relaying a warning never breaks the call
                logger.debug("call unit: could not relay a warning", exc_info=True)
                try:
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
                except Exception:  # noqa: BLE001
                    pass


class CallUnit:
    """Caches one intercepted call against the statement backend.

    This is the runtime counterpart to :func:`call_cache_key`: it builds the
    live inputs that function needs (``arg_digests`` from the actual
    arguments, ``ctx`` from the live processor state), and does the backend
    round-trip -- the same ``backend.get``/``backend.set`` calls the statement
    path uses (:mod:`cash.backends._base`), not a separate KV store.

    **This must never be why user code breaks.** Every failure path -- key
    build, lookup, store -- falls through to calling the original function.
    """

    def __init__(
        self,
        cash_instance,
        ctx_provider: Callable[[], CacheKeyContext],
        loop_vars_provider: Callable[[], dict[str, object]] | None = None,
        loop_var_digests_provider: Callable[[], Mapping[str, str]] | None = None,
        ttl_provider: Callable[[], int | None] | None = None,
        persist_provider: Callable[[], bool] | None = None,
    ):
        self._cash = cash_instance
        #: Builds each call's key and remembers what each site was keyed on.
        self._keys = CallKeys(ctx_provider, loop_vars_provider, loop_var_digests_provider)
        #: Looks call entries up and writes them; holds the sites refused.
        self._entries = CallEntries(cash_instance, ttl_provider, persist_provider)
        self.call_log: list[dict] = []
        #: Per call site, how its calls went in the statement run under way
        #: (see :meth:`_entry_for`); emptied by :meth:`begin_statement`.
        self._site_runs: dict[CallSite, _SiteRun] = {}
        #: What the call just made would have cost to compute: its run time on
        #: a miss, its recorded cost on a hit, ``None`` when it ran plain.
        self._last_compute: float | None = None
        #: Seconds this unit spent on calls beyond their own compute (keys,
        #: lookups, stores, restores), and the compute its hits stood in for.
        #: Monotonic; a statement reads the difference across its run (see
        #: ``StatementProcessor._finish``).
        self.overhead_s = 0.0
        self.hits_saved_s = 0.0
        self._last_hit = False
        self._last_key_s: float | None = None
        self._invoked_keys: list[str | None] = []
        #: The source of the call the current statement is nothing but
        #: (``a, b = build()``), set by the statement with its sites.
        self.plain_value_source: str | None = None
        self.last_returned: tuple[str | None, int, str] | None = None
        #: ``id(result) -> (result, key, digest)`` for the call results this
        #: cell stored or was served, so the statement holding one stores a
        #: reference to its entry rather than a second copy (``call_refs``).
        #: Emptied by :meth:`begin_cell`; holding the results that long keeps
        #: an ``id`` from being reused by another object meanwhile.
        self.held_results: dict[int, tuple[Any, str, str, int]] = {}

    def _cost_floor_s(self) -> float:
        """The bar a call's own execution must clear to be stored (:func:`call_cost_floor_s`)."""
        return call_cost_floor_s(self._cash)

    def begin_cell(self) -> None:
        """A new cell: the results held for references are let go."""
        self.held_results.clear()

    def _hold(self, key: str, value: Any, digest: str | None, size: Any) -> None:
        if digest:
            self.held_results[id(value)] = (value, key, digest, size if isinstance(size, int) else 0)

    def begin_statement(self) -> None:
        """A new statement run: every site starts over (see :meth:`_entry_for`)."""
        self._site_runs.clear()
        self._keys.begin_statement()
        self.last_returned = None

    def outermost_result(self) -> tuple[str, int, str] | None:
        """``(call key, id(result), call source)`` of the call that returned
        LAST in this statement -- in ``x = f(g(y))``, ``f``'s -- or ``None``
        when it ran without the cache. The statement uses it to trust a
        reference without re-digesting the value (``call_refs.with_call_refs``)."""
        found = self.last_returned
        return found if found and found[0] else None

    def _entry_for(self, fn, site: CallSite, invoke):
        """*invoke* behind the many-cheap-calls guard.

        A call in a comprehension is made once per element, and caching one
        costs a key, a lookup, a store and a file tracker of its own: ~14 ms a
        call around a function reading one small file, 5,030 of them in
        ``[read_doc(p) for p in paths]`` -- 8.4 s became 71.6 s. A
        ``for`` loop has ``single_unit_policy.should_run_as_single_unit``
        for exactly this; a comprehension is one statement, so it is decided
        here, by measurement, with the loop's own numbers: past
        ``_GUARD_AFTER_CALLS`` calls in one statement run, if the calls are
        cheap, a few are run plain and timed, and when caching a call costs
        more than ``_OVERHEAD_FACTOR`` times what it computes the rest of
        the statement's calls to this site run plain. Running plain is always
        correct; it is only uncached.
        """
        names: list[str] = []

        def _log_plain(elapsed: float) -> None:
            # Logged as run plain: left out, the badge's count of a site's
            # calls came up short by every call the guard ran without the
            # cache -- 5220/5225 for a folder of 5,225 files.
            if not names:
                names.append(self._func_name(fn))
            self._record(names[0], site, None, cache_hit=False, elapsed=elapsed, ran_plain=True)

        @functools.wraps(fn)
        def _entry(*args, **kwargs):
            __tracebackhide__ = True  # noqa: F841 - see _guarded
            with _warnings_at_the_caller():
                return _guarded(*args, **kwargs)

        def _guarded(*args, **kwargs):
            # IPython leaves a frame with this set out of the traceback it
            # prints: a user's KeyError showed three of cash's wrapper frames
            # between their cell and their function.
            __tracebackhide__ = True  # noqa: F841
            run = self._site_runs.get(site)
            if run is None:
                run = self._site_runs[site] = _SiteRun()
            if run.plain:
                started = _perf_counter()
                result = fn(*args, **kwargs)
                _log_plain(_perf_counter() - started)
                self.last_returned = (None, id(result), site.source)
                return result
            if run.probing:
                started = _perf_counter()
                result = fn(*args, **kwargs)
                run.plain_s += _perf_counter() - started
                _log_plain(_perf_counter() - started)
                run.plain_n += 1
                if run.plain_n >= _PLAIN_SAMPLES:
                    run.probing = False
                    cached = run.total_s / run.calls
                    plain = run.plain_s / run.plain_n
                    keyed = run.key_s / run.calls
                    # Too dear to cache, or a hit could not save a quarter of
                    # the call: a hit pays the key and lookup, then the restore.
                    run.plain = cached > (1 + _OVERHEAD_FACTOR) * plain or keyed >= _HIT_MUST_SAVE * plain
                    run.decided = True
                    trace_event(
                        "call_site_decided",
                        source=site.source,
                        calls=run.calls,
                        cached_ms=round(cached * 1000, 3),
                        keyed_ms=round(keyed * 1000, 3),
                        plain_ms=round(plain * 1000, 3),
                        plain=run.plain,
                    )
                self.last_returned = (None, id(result), site.source)
                return result
            self._last_compute = None
            self._last_key_s = None
            self._last_hit = False
            # One slot per invocation: a call the callee makes through a
            # lambda it was handed runs its own `_invoke` inside this one.
            self._invoked_keys.append(None)
            started = _perf_counter()
            try:
                result = invoke(*args, **kwargs)
            finally:
                invoked_key = self._invoked_keys.pop()
            spent = _perf_counter() - started
            self.last_returned = (invoked_key, id(result), site.source)
            run.total_s += spent
            if self._last_compute is not None:
                if self._last_hit:
                    self.overhead_s += spent
                    self.hits_saved_s += self._last_compute
                else:
                    self.overhead_s += max(0.0, spent - self._last_compute)
            run.calls += 1
            if self._last_key_s is not None:
                run.key_s += self._last_key_s
            if self._last_compute is not None:
                run.compute_s += self._last_compute
                run.computed += 1
            if (
                not run.decided
                and run.calls >= _GUARD_AFTER_CALLS
                and run.computed
                and run.compute_s / run.computed < _GUARD_CHEAP_BELOW_S
            ):
                run.probing = True
            return result

        return _entry

    def wrap(self, fn, site: CallSite):
        func_name = self._func_name(fn)

        def _invoke(*args, **kwargs):
            __tracebackhide__ = True  # noqa: F841 - see _entry_for
            # Globals this callee writes. Resolved per call rather
            # than once per `wrap`, because the underlying source analysis is
            # memoised (`callee_mutated_globals`) while the "is it bound, is it
            # a module" filter genuinely depends on the live namespace. Empty
            # for nearly every callee, and every branch below short-circuits on
            # empty, so an ordinary call pays one memo lookup.
            key_started = _perf_counter()
            mutated_globals = callee_mutated_globals(fn)
            key = self._keys.key(
                site,
                args,
                kwargs,
                global_digests(fn, mutated_globals) if mutated_globals else None,
                # A callee that writes globals keys on their state: the old way.
                fn=None if mutated_globals else fn,
            )
            if key is None:
                # Either the key build raised, or `call_cache_key` itself
                # refused (arg-digest count mismatch, by design). Either way
                # this call is not safely keyable right now -- run it
                # uncached rather than risk a collapsed, wrong key.
                return fn(*args, **kwargs)

            # Only a result this entry holds may be referred to: set once the
            # key is usable, and never for a refused key below.
            if self._invoked_keys:
                self._invoked_keys[-1] = key
            if self._entries.is_refused(key):
                if self._invoked_keys:
                    self._invoked_keys[-1] = None
                # A previous miss on this exact site proved its effects
                # cannot be replayed (argument mutation or an RNG draw). Run
                # it plain -- never look it up, never store over it.
                return fn(*args, **kwargs)

            hit_started = _perf_counter()
            hit, value, recorded_cost, metadata = self._entries.lookup(key)
            self._last_key_s = _perf_counter() - key_started
            if hit:
                value, captured_globals = unwrap_callee_globals(value, metadata)
                if value is UNWRAP_FAILED:
                    # The entry says it carries captured globals and does not.
                    # Treat it as absent rather than hand back a tuple where a
                    # value belongs -- a miss costs a recompute, this would be
                    # a silently wrong value.
                    return fn(*args, **kwargs)
                # Replay the ORIGINAL execution's observations into the
                # statement's ambient capture (FileAccessTracker, live
                # stdout/stderr). The call itself does not run on a hit, so
                # without this replay the tracker records no read and the
                # live stream sees no print -- the enclosing statement's own
                # entry would then be rewritten (on ITS next miss) from a
                # degraded observation that is missing both. Mirrors
                # ``core.py``'s ``_propagate_file_deps_to_active_tracker``,
                # the ``@cash.cache`` decorator's defence against the same
                # failure mode.
                replay_deps(metadata)
                replay_output(metadata)
                restore_globals(fn, mutated_globals, captured_globals)
                if not captured_globals:
                    self._hold(key, value, metadata.get(DIGEST_FIELD), metadata.get(SIZE_FIELD))
                self._record(func_name, site, key, cache_hit=True, elapsed=0.0, time_saved=recorded_cost)
                self._last_compute = recorded_cost or 0.0
                self._last_hit = True
                self._entries.drop_if_hit_costs_more(key, _perf_counter() - hit_started, recorded_cost)
                return value

            # The call runs inside the STATEMENT's ambient capture
            # (FileAccessTracker, RNG capture, output capture), which wraps
            # the whole statement's exec. On a genuine miss that capture
            # records this call's effects as its own, for free, and
            # everything is correct. The broken case is a later run where the
            # STATEMENT misses and re-executes but the CALL hits: the call's
            # effects do not re-happen, so the statement's own capture is
            # rewritten from a degraded observation. Both checks below fail
            # CLOSED -- refuse the site outright rather than serve a value
            # whose side effects will not re-happen.
            #
            # File deps and stdout/stderr are recorded around the call so a
            # LATER hit can replay them (above). RNG is deliberately not
            # recorded here -- a call that consumed it is already refused.
            #
            # A NESTED tracker, not a before/after diff against the ambient
            # (statement-wide) one -- ``core.py:1870`` is what this task was
            # told to copy, and it wraps the DECORATED CALL in its own fresh
            # ``FileAccessTracker(propagate_to_parent=True)``, not a diff
            # against the caller's tracker. That distinction is load-bearing:
            # ``active_tracker.get()`` is shared for the whole statement (or,
            # inside a loop, the whole loop-as-one-unit execution), so a diff
            # against it goes silently wrong the moment the SAME path is read
            # twice in one tracker window -- ``hdr = read(p); total =
            # expensive(k)``, or two loop iterations both reading the same
            # file. The second read's "after" set already contains the path
            # from the first, so ``after - before`` is EMPTY: the entry
            # stores no dependency at all, and ``CallEntries._auto_file_deps_fresh`` is
            # vacuously true forever. A fresh, per-call tracker has no such
            # baseline to collide with -- its own set IS this call's reads,
            # full stop -- and ``propagate_to_parent=True`` still surfaces
            # every read to the enclosing statement's tracker immediately, so
            # the miss-path "recorded for free" behaviour is unchanged.
            rng_before = capture_rng_state()
            arg_hashes_before = hash_args(args, kwargs)
            started = _perf_counter()
            call_tracker = FileAccessTracker(
                getattr(fn, "__globals__", None),
                propagate_to_parent=True,
            )
            with call_tracker:
                result, stdout_text, stderr_text = call_capturing_output(fn, args, kwargs)
            elapsed = _perf_counter() - started
            stored = False

            if rng_modules_changed(rng_before, capture_rng_state()):
                # RNG is a consumed linear resource -- what matters is stream
                # POSITION, not membership. A hit leaves the global stream
                # where it was, so every downstream draw would diverge from
                # the uncached oracle. Replaying it properly needs a
                # sub-statement position anchor that does not exist yet;
                # v1 refuses instead of guessing.
                self._entries.refuse(key)
            elif hash_args(args, kwargs) != arg_hashes_before:
                # The callee mutated a live argument in place and returned
                # something else (`df.dropna(inplace=True); return len(df)`).
                # The identity check only catches `return arg` -- this
                # catches "mutated but returned a *different* object", which
                # a hit would silently skip.
                self._entries.refuse(key)
            elif (
                elapsed >= self._cost_floor_s()
                and self._entries.storable(result, args, kwargs)
                and self._entries.restore_pays(result, elapsed)
            ):
                # The callee's writes to its own globals, captured as
                # an END STATE. Snapshotting the final value needs no ordering
                # and no idempotence, which is why this is tractable where
                # replaying the individual mutations is not.
                #
                # `None` means "cannot capture this soundly" -- an unpicklable
                # value, or one whose hash falls back to identity so a later
                # pre-state comparison could not tell it had changed. Refuse
                # the SITE rather than store an entry whose restore would be
                # wrong, matching the RNG and argument-mutation refusals above:
                # an uncached call is merely slow.
                captured = capture_globals(fn, mutated_globals)
                if captured is None:
                    self._entries.refuse(key)
                else:
                    held = self._entries.store(
                        key,
                        result,
                        elapsed,
                        file_deps=frozenset(call_tracker.get_accessed_files()),
                        remote_deps=frozenset(call_tracker.get_accessed_remote_urls()),
                        stdout=stdout_text,
                        stderr=stderr_text,
                        callee_globals=captured,
                        function=func_name,
                        plain_value=site.source == self.plain_value_source,
                        code_module=getattr(fn, "__module__", None),
                    )
                    if held:
                        self._hold(key, result, *held)
                    stored = True
            self._record(func_name, site, key, cache_hit=False, elapsed=elapsed, stored=stored)
            self._last_compute = elapsed
            return result

        return self._entry_for(fn, site, _invoke)

    def _func_name(self, fn) -> str:
        """The name this call's events display under in the badge and stats.

        Delegates to ``Cash.get_func_key`` for a stable ``module.qualname``
        rather than rebuilding the rule here.
        """
        try:
            return self._cash.get_func_key(fn)
        except Exception:  # noqa: BLE001
            return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"

    def _record(
        self, func_name, site: CallSite, key, *, cache_hit, elapsed, time_saved=0.0, ran_plain=False, stored=True
    ) -> None:
        """Emit the SAME event shape ``drain_decorator_calls`` returns.

        Keeping the contract identical is what lets the badge, the ``@cache``
        row and ``%cash_stats`` keep working on this log unmodified.

        ``intercepted`` is set to ``True`` unconditionally: every event this
        class emits is, by construction, one ``CallCache.resolve`` routed
        through the interception path. There is no name-reconciliation step
        to keep in sync with the badge any more (see the note on
        ``CallCache.__init__`` where ``wrapped_names`` used to live).
        """
        self.call_log.append(
            {
                "func_name": func_name,
                "cache_hit": cache_hit,
                "execution_time": elapsed,
                "time_saved": time_saved,
                "args_hash": "",
                "cache_key": key,
                "timestamp": _time.time(),
                "call_source": site.source,
                "occurrence_index": site.occurrence_index,
                "intercepted": True,
                # Run without the cache by the many-cheap-calls guard.
                "ran_plain": ran_plain,
                # A miss whose result went to the cache. False for a call below
                # the cost floor or refused: the badge has nothing to say about it.
                "stored": bool(stored),
                # Why this call was not served: which part of its key moved since
                # the site was last keyed. Only on a miss, and only when known.
                "miss_reason": None if cache_hit else self._keys.why_missed(site),
            }
        )

    def drain(self) -> list[dict]:
        events, self.call_log = self.call_log, []
        return events
