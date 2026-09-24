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
import copy as _copy
import dataclasses
import functools
import logging
import pathlib as _pathlib
import sys
import time as _time
import uuid
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from cash._clock import perf_counter as _perf_counter
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability, identity_coupled_reason
from cash.backends._base import ttl_expired
from cash.backends.value_policy import worth_its_bytes
from cash.notebook._trace import trace_event
from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import CallSite, names_read
from cash.notebook.call_key import CallKeys, callee_mutated_globals, global_digests
from cash.notebook.call_refs import (
    DIGEST_FIELD,
    ESTIMATED_FIELD,
    SIZE_FIELD,
    UNHASHED_PREFIX,
    digest_and_size,
)
from cash.notebook.consumables import is_consumable_unrestorable
from cash.object_hashing import (
    compute_hash,
    estimate_object_size,
    is_identity_fallback_hash,
    pickled_size_estimate,
)
from cash.tracking.file_dep_snapshot import (
    attach_code_relative,
    dep_path_for_this_process,
    snapshot_dependencies,
    snapshot_is_fresh,
)
from cash.tracking.file_tracker import FileAccessTracker
from cash.tracking.randomness import capture_rng_state, rng_modules_changed
from cash.tracking.tracker_context import active_tracker

from ..cost_model import estimated_restore_time
from ._tee import TeeWriter

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


#: Returned by :func:`_unwrap_callee_globals` when an entry claims to carry a
#: callee's captured globals but does not have the shape to prove it. A unique
#: sentinel rather than ``None``, because ``None`` is a perfectly good cached
#: value and must stay distinguishable from a broken entry.
_UNWRAP_FAILED = object()


def _unwrap_callee_globals(value, metadata: Mapping[str, Any]):
    """Split a stored value into ``(result, captured_globals)``.

    Entries that captured a callee's writes to globals store
    ``(result, {name: value})`` and set a plain ``has_callee_globals`` bool in
    metadata; every other entry stores the bare result. The flag makes the
    shape self-describing rather than something the reader has to guess from
    the value's type -- a cached call that legitimately returns a 2-tuple would
    otherwise be indistinguishable from a wrapped one.

    Returns ``(_UNWRAP_FAILED, None)`` when the flag is set and the shape does
    not match. That can only mean a corrupt or hand-edited entry, and handing
    back the tuple as though it were the result would be a silently wrong
    value where a miss merely costs a recompute.
    """
    if not metadata.get("has_callee_globals"):
        return value, None
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        return value[0], value[1]
    logger.debug("call unit: entry claims captured globals but has the wrong shape")
    return _UNWRAP_FAILED, None


#: The many-cheap-calls guard (``CallUnit._entry_for``), in the numbers
#: ``for_handler`` uses to run a loop as one unit: past 50 calls in one
#: statement run, calls cheaper than this are timed plain on a few samples, and
#: the site runs plain for the rest of the run when caching a call costs more
#: than ``_OVERHEAD_FACTOR`` times what the call computes.
_GUARD_AFTER_CALLS = 50
#: A call cheaper than this gets no content digest, so no statement refers to it.
_REF_MIN_COMPUTE_S = 0.1
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


#: Result types whose identity no program can rely on -- see
#: ``CallUnit._storable``. Exact types only: a subclass may carry state.
_IDENTITY_FREE = frozenset({int, float, complex, bool, str, bytes, type(None)})


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
        # The TTL in force for the statement this call sits in.
        # Read at INVOKE time, like `loop_vars_provider`, because one
        # `CallCache` serves every statement and each brings its own
        # annotation. `None` -- the default, and what every direct
        # construction predating this parameter gets -- means "no TTL", which
        # is the behaviour this class had when it ignored TTL entirely.
        self._ttl_provider = ttl_provider or (lambda: None)
        # `# @cash:persist` / `%cash_persist` for the statement this call sits
        # in. Read at STORE time for the same reason `ttl` is read at
        # invoke time: one `CallCache` serves every statement. `False` -- the
        # default, and what every construction predating this parameter gets --
        # leaves the decision to the cost model, which is what this class did
        # when it ignored `persist` entirely.
        self._persist_provider = persist_provider or (lambda: False)
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
        #: Cache keys of sites known to mutate an argument or consume RNG,
        #: discovered by observing a MISS (see `wrap`). Permanent for the life
        #: of this `CallUnit` (one notebook session): once a site is known to
        #: have an effect this feature cannot replay, it must never be served
        #: or written again, on any later call -- including one wrapped by a
        #: fresh `wrap()` call for the same site (`CallCache.resolve` re-wraps
        #: per statement execution), which is why this lives on `self` and not
        #: on `_invoke`'s closure.
        self._refused: set[str] = set()

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
            if key in self._refused:
                if self._invoked_keys:
                    self._invoked_keys[-1] = None
                # A previous miss on this exact site proved its effects
                # cannot be replayed (argument mutation or an RNG draw). Run
                # it plain -- never look it up, never store over it.
                return fn(*args, **kwargs)

            hit_started = _perf_counter()
            hit, value, recorded_cost, metadata = self._lookup(key)
            self._last_key_s = _perf_counter() - key_started
            if hit:
                value, captured_globals = _unwrap_callee_globals(value, metadata)
                if value is _UNWRAP_FAILED:
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
                self._replay_deps(metadata)
                self._replay_output(metadata)
                self._restore_globals(fn, mutated_globals, captured_globals)
                if not captured_globals:
                    self._hold(key, value, metadata.get(DIGEST_FIELD), metadata.get(SIZE_FIELD))
                self._record(func_name, site, key, cache_hit=True, elapsed=0.0, time_saved=recorded_cost)
                self._last_compute = recorded_cost or 0.0
                self._last_hit = True
                self._drop_if_hit_costs_more(key, _perf_counter() - hit_started, recorded_cost)
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
            # stores no dependency at all, and ``_auto_file_deps_fresh`` is
            # vacuously true forever. A fresh, per-call tracker has no such
            # baseline to collide with -- its own set IS this call's reads,
            # full stop -- and ``propagate_to_parent=True`` still surfaces
            # every read to the enclosing statement's tracker immediately, so
            # the miss-path "recorded for free" behaviour is unchanged.
            rng_before = capture_rng_state()
            arg_hashes_before = self._hash_args(args, kwargs)
            started = _perf_counter()
            call_tracker = FileAccessTracker(
                getattr(fn, "__globals__", None),
                propagate_to_parent=True,
            )
            with call_tracker:
                result, stdout_text, stderr_text = self._call_capturing_output(fn, args, kwargs)
            elapsed = _perf_counter() - started
            stored = False

            if rng_modules_changed(rng_before, capture_rng_state()):
                # RNG is a consumed linear resource -- what matters is stream
                # POSITION, not membership. A hit leaves the global stream
                # where it was, so every downstream draw would diverge from
                # the uncached oracle. Replaying it properly needs a
                # sub-statement position anchor that does not exist yet;
                # v1 refuses instead of guessing.
                self._refused.add(key)
            elif self._hash_args(args, kwargs) != arg_hashes_before:
                # The callee mutated a live argument in place and returned
                # something else (`df.dropna(inplace=True); return len(df)`).
                # The identity check only catches `return arg` -- this
                # catches "mutated but returned a *different* object", which
                # a hit would silently skip.
                self._refused.add(key)
            elif (
                elapsed >= self._cost_floor_s()
                and self._storable(result, args, kwargs)
                and self._restore_pays(result, elapsed)
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
                captured = self._capture_globals(fn, mutated_globals)
                if captured is None:
                    self._refused.add(key)
                else:
                    self._store(
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
                    stored = True
            self._record(func_name, site, key, cache_hit=False, elapsed=elapsed, stored=stored)
            self._last_compute = elapsed
            return result

        return self._entry_for(fn, site, _invoke)

    def _call_capturing_output(self, fn, args: tuple, kwargs: dict) -> tuple[Any, str, str]:
        """Run *fn*, returning ``(result, stdout_text, stderr_text)``.

        Tees ``sys.stdout``/``sys.stderr`` through a recorder that still
        forwards every byte to the stream that was live going in -- which,
        during a real statement execution, IS the statement's own ambient
        capture (a ``StringIO``, a ``TeeWriter``, or the real terminal
        outside any capture). So a genuine miss looks exactly as it did
        before this method existed: the callee's output reaches the
        statement's capture "for free", untouched.
        The recording exists only so a LATER hit (:meth:`_replay_output`) has
        something to write back onto the live stream -- otherwise that
        output is simply gone, since the callee does not run at all on a hit.
        """
        __tracebackhide__ = True  # noqa: F841 - see _entry_for
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee_out = TeeWriter(old_stdout)
        tee_err = TeeWriter(old_stderr)
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            result = fn(*args, **kwargs)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        return result, tee_out.getvalue(), tee_err.getvalue()

    def _replay_deps(self, metadata: Mapping[str, Any]) -> None:
        """Re-declare a hit entry's recorded file/remote deps as though THIS
        call had just read them, onto the statement's ambient tracker.

        Attribution AND propagation: the call unit already validated these
        deps before serving the hit (they are checked as part of the
        entry's own freshness -- a stale file behind ``key`` simply misses,
        same as the statement path), and the enclosing statement still needs
        them registered because its own cached value transitively depends on
        the same files. Without this, a statement that only reaches a file
        through a now-cached sub-call would lose that dependency the moment
        the sub-call started hitting -- exactly the regression this
        closes.
        """
        snap = metadata.get("auto_file_deps")
        if not snap:
            return
        try:
            tracker = active_tracker.get()
        except Exception:  # noqa: BLE001 - tracking is best-effort
            return
        if tracker is None:
            return
        for path, recorded in snap.items():
            try:
                # A remote entry must go back onto the remote channel --
                # routed to ``add_tracked`` it would enter the file set, be
                # stat'ed, and be dropped, same reasoning as
                # ``core.py``'s ``_propagate_file_deps_to_active_tracker``.
                if isinstance(recorded, dict) and recorded.get("remote"):
                    tracker.add_tracked_remote(path)
                else:
                    # The file THIS process reads, as the decorator replays it.
                    tracker.add_tracked(dep_path_for_this_process(path, recorded))
            except Exception:  # noqa: BLE001
                logger.debug("call unit: could not replay dep %r", path)

    def _replay_output(self, metadata: Mapping[str, Any]) -> None:
        """Write a hit entry's recorded stdout/stderr onto the LIVE stream.

        The callee did not run this time, so its prints never happened;
        writing the recorded text to ``sys.stdout``/``sys.stderr`` puts it
        back wherever the statement's ambient capture currently points (a
        buffer during a real run, the real terminal in a bare unit test),
        which is what reconstructs ``print(a); f(x); print(b)``'s
        interleaving without the statement path needing to know sub-call
        caching exists at all.
        """
        stdout_text = metadata.get("stdout") or ""
        stderr_text = metadata.get("stderr") or ""
        if stdout_text:
            try:
                sys.stdout.write(stdout_text)
            except (OSError, ValueError, TypeError, AttributeError):  # a closed or foreign stream
                logger.debug("call unit: could not replay stdout", exc_info=True)
        if stderr_text:
            try:
                sys.stderr.write(stderr_text)
            except (OSError, ValueError, TypeError, AttributeError):  # a closed or foreign stream
                logger.debug("call unit: could not replay stderr", exc_info=True)

    def _hash_args(self, args: tuple, kwargs: dict) -> tuple:
        """Content hashes of the live arguments, for mutation detection.

        Uses the sampling hash (`compute_hash`) deliberately, not
        `compute_hash_full`: it is the same one the statement path's own
        content observation uses, and for a large frame a full hash per call
        would cost more than the call being cached is worth.

        Two DIFFERENT ways this can under-report a mutation, and they get
        different treatment:

        1. **Sampling.** `compute_hash` samples large objects (ndarray: first
           100 elements, DataFrame: first 5 rows, collections >200: head/tail).
           A same-size in-place edit outside the sampled region is invisible
           here. This is a known, accepted trade -- it errs toward CACHING for
           objects that still hash BY CONTENT, and the identity check in
           `_storable` stays as a second line of defence for the one shape it
           fully covers (`return arg`).

        2. **Identity fallback.** `compute_hash`'s tier 3
           (`object_hashing.identity_hash`) hashes `id(obj)`, not the object's
           data, once pickling itself has failed (a `threading.Lock`, a socket,
           an open file, anything with an unpicklable `__reduce__`). `id(obj)`
           is invariant across an in-place mutation of that SAME object, so
           this is not "a coarser content hash" the way sampling is -- it is
           BLIND to every mutation of that argument, always, for the entire
           unpicklable-object class. Comparing two such hashes before/after a
           call would silently read as "unchanged" even when the callee
           mutated the object, which directly contradicts "fail closed": a
           value flagged via `is_identity_fallback_hash` is therefore replaced
           with a fresh, single-use sentinel (`object()`) instead of the hash
           string. Two distinct `object()` instances are never `==`, so the
           before/after tuple comparison in `wrap` always reads as "changed"
           for that argument -- i.e. "cannot prove this argument is clean" is
           treated the same as "proved it changed", which is the fail-closed
           direction the task requires.
        """
        out = []
        for value in (*args, *kwargs.values()):
            try:
                h = compute_hash(value)
            except Exception:  # noqa: BLE001
                # This branch IS live, on every Python before 3.14: hashing an
                # instance of a locally-defined class raises
                # `AttributeError: Can't pickle local object '<f>.<locals>.C'`
                # rather than reaching `compute_hash`'s identity fallback. A
                # class defined inside a function is ordinary in a notebook and
                # ubiquitous in tests.
                #
                # It must append the SAME single-use sentinel as the
                # identity-fallback case below, and for the same reason. This
                # used to append `None`, on the reasoning that "'cannot prove
                # unmutated' is exactly what a `None` here already means to the
                # caller" -- but `None == None`, so two unknowable snapshots
                # compared EQUAL and read as "argument unchanged". That is
                # fail-OPEN: a callee mutating an unpicklable argument was
                # cached and its mutation silently skipped, on 3.10-3.13.
                out.append(object())
                continue
            if is_identity_fallback_hash(value, h):
                out.append(object())
            else:
                out.append(h)
        return tuple(out)

    @staticmethod
    def _capture_globals(fn, names: tuple[str, ...]) -> dict[str, Any] | None:
        """Post-call values of the globals *fn* writes, or ``None`` to refuse.

        ``None`` is returned when any watched name cannot be captured soundly:

        * **absent** — it was there when the watch list was filtered and is not
          now, so this call's effect on it cannot be described;
        * **identity-fallback hash** — ``compute_hash`` fell through to
          ``sha256(str(id(obj)))`` because the object does not pickle (a lock,
          a socket, an open file). ``id`` is invariant across an in-place
          mutation, so a later key comparison on this name is BLIND, always,
          for that whole class. Storing an entry whose pre-state cannot be
          told apart is exactly the partial-accumulator hazard.

        Returning ``None`` costs a permanently-uncached site. Storing anyway
        would cost a silently wrong restore, and this method exists to prefer
        the former.

        **Deep-copied, not referenced.** The whole point of this capture is a
        POST-CALL snapshot, and the object being snapshotted is by construction
        one that gets mutated in place -- so keeping a reference does not
        capture a state at all, it captures a live handle that keeps changing.
        The RAM tier stores metadata as given, so a later call mutating the
        same object silently rewrites an already-stored entry's recorded
        "post-state". Measured::

            cell 3   a = next_seq()            stores N -> [1]
            cell 5   seen.append(next_seq())   mutates the SAME list to [2]
            rerun    cell 3 hits, restores N -> [2]   (not [1])

        which then re-keyed cell 5's call against a pre-state that had never
        existed, so it missed forever -- and the two spellings diverged.
        A copy failure is treated like any other
        "cannot capture this soundly": refuse.
        """
        if not names:
            return {}
        globals_dict = getattr(fn, "__globals__", None) or {}
        captured: dict[str, Any] = {}
        for name in names:
            if name not in globals_dict:
                return None
            value = globals_dict[name]
            try:
                if is_identity_fallback_hash(value, compute_hash(value)):
                    return None
                captured[name] = _copy.deepcopy(value)
            except Exception:  # noqa: BLE001 - cannot prove it is capturable
                return None
        return captured

    def _restore_globals(self, fn, names: tuple[str, ...], recorded: Mapping[str, Any] | None) -> None:
        """Land a hit entry's recorded post-call globals back into *fn*'s own
        namespace, so the callee's write survives a call it did not make.

        The counterpart of :meth:`_replay_output` for state rather than text,
        and the call-level twin of what the statement path does by listing the
        same names in its ``outputs``.

        A rebind, not an in-place transfer. That matches the statement path's
        default restore (``StatementRestorer._write_restored_value`` only
        transfers in place for an explicitly-listed estimator-fit receiver), so
        the two spellings of the same code land the value the same way. An
        alias taken BEFORE the restore therefore keeps pointing at the old
        object — a real limitation, and the same one the statement path has
        always had for every restored variable.

        Only names in *names* are written. The entry could carry a stale name
        from a since-edited callee, and honouring it would resurrect a variable
        the current source never mentions.
        """
        if not names:
            return
        if not isinstance(recorded, Mapping) or not recorded:
            return
        globals_dict = getattr(fn, "__globals__", None)
        if not isinstance(globals_dict, dict):
            return
        for name in names:
            if name in recorded:
                # A COPY, for the mirror of the reason `_capture_globals`
                # copies: handing back the stored object would make the live,
                # about-to-be-mutated variable and the cache entry the same
                # object, so the next call would rewrite the entry it was just
                # served from.
                try:
                    globals_dict[name] = _copy.deepcopy(recorded[name])
                except Exception:  # noqa: BLE001 - a restore must never crash
                    logger.debug("call unit: could not restore global %r", name)

    #: A hit is judged a loss only past this, so timer noise on a cheap call
    #: never refuses it.
    _MIN_HIT_LOSS_S = 0.05

    def _restore_pays(self, result, elapsed: float) -> bool:
        """Whether restoring *result* is predicted to beat computing it again.

        The statement path has refused a value whose predicted restore exceeds
        80% of its compute since the cost model was fitted; the call cache,
        which holds the biggest values in a notebook,
        never asked. Predicted for disk -- where it
        comes back from after a restart, the case a cache is for.
        """
        try:
            size = estimate_object_size(result)
            predicted = estimated_restore_time(type(result).__name__, size, "disk")
        except Exception:  # noqa: BLE001 - no prediction: store, as before
            return True
        return predicted <= max(self._MIN_HIT_LOSS_S, 0.8 * elapsed)

    def _drop_if_hit_costs_more(self, key: str, hit_cost: float, saved: float) -> None:
        """A hit that took longer than the compute it saved is a loss: stop.

        Measured, not predicted -- a prediction can be wrong for a type it was
        not fitted on, and one sweep's hits were ~10 s against ~4 s of compute,
        reported as "4/4 hit". The entry is dropped and the site runs plain
        for the rest of the session (`_refused`, the same bench the argument-
        mutation and RNG refusals use), so the next run computes rather than
        paying the loss again.
        """
        if hit_cost <= max(self._MIN_HIT_LOSS_S, saved or 0.0):
            return
        self._refused.add(key)
        try:
            self._cash.backend.delete(key)
        except Exception:  # noqa: BLE001 - reclaiming is best effort; the refusal holds
            pass
        logger.debug(
            "[CALL_UNIT] hit on %s took %.2fs to save %.2fs: dropped, runs plain", key[:16], hit_cost, saved or 0.0
        )

    def _storable(self, result, args, kwargs) -> bool:
        """Refuse values whose *identity* is load-bearing.

        Three families, all of which the statement path already refuses in its
        own vocabulary:

        1. **The result IS one of the arguments.** ``def f(d): d['k']=1;
           return d`` -- a hit would hand back a deserialised copy, so
           ``a = f(d)`` gives ``a is not d`` where Python guarantees identity.
           The statement path's alias rule only reaches a bare bind
           (``b = a``); the computed-RHS version is
           structurally unfixable per-statement. At the call node the live
           arguments are in hand, so it is one ``is`` check.

           Not for a plain scalar. CPython shares one object for small ints
           and interned strings, so ``score(1, 10)`` returns the very ``10``
           it was passed -- the check refused that call on every run, and it
           was always the first iteration of a sweep that re-ran. No program
           can rely on the identity of an int or a str.

        2. **Identity-coupled library objects** -- a matplotlib Figure/Axes is
           only correct while it IS the object pyplot's registry points at.
           The RAM tier deep-copies on store and ``Figure.__setstate__``
           re-registers the COPY as the current figure, so a later bare
           ``plt.savefig()`` writes the cache's snapshot. Refusing here lands
           BEFORE the write, which is what stops the copy being made at all.

        3. **A consumable the store cannot copy** -- an open file, a
           generator. The RAM tier keeps it by reference, so a hit hands back
           the very object a reader already drained. The statement path
           refuses such an output for the same reason.

        A caching optimisation must never be why user code fails. The two
        ``is`` loops above cannot themselves raise -- identity comparison
        never does -- so the only place this can fail is the
        ``identity_coupled_reason`` call, guarded below. Refusing to store is
        free (the call just runs uncached next time); wrongly storing is not
        (it is exactly the silent-wrong-answer / hijacked-identity bug this
        method exists to prevent), which argues for failing toward ``False``.
        But ``identity_coupled_reason`` is pure MRO-qualname introspection --
        by design it never imports matplotlib and has no I/O -- so this
        except is a belt no realistic value should ever reach; returning
        ``True`` here mirrors the already-shipped fallback in
        ``call_interception._is_storable`` (same delegation, same except
        clause) so a call's storability does not silently depend on which of
        the two dispatch paths happened to route it.
        """
        if type(result) not in _IDENTITY_FREE:
            for arg in args:
                if result is arg:
                    return False
            for arg in kwargs.values():
                if result is arg:
                    return False
        try:
            return identity_coupled_reason("<intercepted call>", result) is None and not is_consumable_unrestorable(
                result
            )
        except Exception:  # noqa: BLE001 - never let the predicate break the call
            return True

    def _lookup(self, key: str) -> tuple[bool, Any, float, dict]:
        """``(hit, value, recorded_execution_time, metadata)`` -- one backend read.

        ``backend.get`` returns ``(metadata, value)`` (``cash.backends._base``);
        ``metadata is None`` is the key-presence test the statement path itself
        uses (``CacheFreshnessChecker.check_cache``), since a stored ``None``
        value is still a legitimate hit. *metadata* is returned too (rather
        than just the cost pulled out of it) so the caller can replay the
        file/remote/stdout/stderr channels this entry recorded -- see
        :meth:`_replay_deps` / :meth:`_replay_output`. Backends round-trip
        metadata as an opaque plain ``dict`` (see ``CacheMetadata``'s
        docstring in ``backends/_base.py``); a non-mapping value is treated
        defensively as empty rather than trusted.

        **A key match alone is not enough to call this a hit.** A call's
        cache KEY carries source + argument/loop-var lineage -- never file
        content -- so a stored entry whose recorded file read has since
        changed on disk would otherwise be served forever, regardless of
        this task's dependency-propagation fix: propagating a dependency the
        call itself never re-checks would just make the STATEMENT re-declare
        a staleness nobody underneath it ever notices. ``_auto_file_deps_fresh``
        re-validates it through ``snapshot_is_fresh``, the check
        ``Cash._auto_file_deps_fresh`` makes -- a stale entry is treated as a miss like any other,
        so it falls through to a genuine recompute (and gets overwritten
        under the same key) rather than being replayed.
        """
        try:
            metadata, value = self._cash.backend.get(key)
        except Exception:  # noqa: BLE001
            return False, None, 0.0, {}
        if metadata is None:
            return False, None, 0.0, {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        if not self._auto_file_deps_fresh(metadata):
            return False, None, 0.0, {}
        if not self._ttl_fresh(metadata):
            return False, None, 0.0, {}
        try:
            cost = float(metadata.get("execution_time", 0.0))
        except (TypeError, ValueError, AttributeError):
            cost = 0.0
        return True, value, cost, metadata

    def _ttl_fresh(self, metadata: Mapping[str, Any]) -> bool:
        """The statement's TTL applied to a call entry, by the rule
        every cache path shares (:func:`cash.backends._base.ttl_expired`).

        Before this, ``call_unit.py`` contained no reference to ``ttl`` at all,
        so call entries never expired. Once call interception became the
        default that quietly hollowed out the annotation: the
        STATEMENT would expire and re-execute while the expensive call inside
        it was still served from an entry with no expiry. Measured on
        ``# @cash:ttl=0`` -- the spelling the docs give for data that must
        never be served stale -- the work did not re-run at all until
        ``# @cash:no-cache-calls`` was added as well.
        """
        try:
            timestamp = float(metadata.get("timestamp") or 0)
        except (TypeError, ValueError):
            timestamp = 0.0
        return not ttl_expired(timestamp, self._ttl_provider())

    @staticmethod
    def _auto_file_deps_fresh(metadata: Mapping[str, Any]) -> bool:
        """Is every dependency the call recorded still as it was?

        The same snapshot shape and the same :func:`snapshot_is_fresh` the
        decorator uses, so the two cannot drift on what "fresh" means -- or on
        where a file beside the callee's own code is looked for. Absent/empty
        ``auto_file_deps`` (a call that read no files) is vacuously fresh.
        """
        try:
            fresh, _stale = snapshot_is_fresh(metadata.get("auto_file_deps"))
        except Exception:  # noqa: BLE001 - fail closed: cannot prove fresh
            return False
        return fresh

    def _store(
        self,
        key: str,
        value,
        elapsed: float,
        *,
        file_deps: frozenset[str] = frozenset(),
        remote_deps: frozenset[str] = frozenset(),
        stdout: str = "",
        stderr: str = "",
        callee_globals: Mapping[str, Any] | None = None,
        function: str | None = None,
        plain_value: bool = False,
        code_module: str | None = None,
    ) -> None:
        """Write through ``backend.set(key, value, metadata)`` -- the same
        two-positional-argument shape the statement path uses
        (``StatementStore``), not a merged single-dict entry.

        ``file_deps``/``remote_deps`` are snapshotted (mtime/size/hash, or a
        remote validator token) into ONE ``auto_file_deps`` dict -- the exact
        field name and shape ``Cash``'s own decorator writes
        (``_snapshot_tracked_deps`` in ``core.py``) -- rather than two bare
        path lists. A bare list has nothing for :meth:`_auto_file_deps_fresh`
        to compare against; the snapshot is what makes this call's OWN hit
        path able to notice the file it read has since changed, not just
        propagate a dependency nobody re-checks.
        ``stdout``/``stderr`` are omitted when empty, same as
        ``auto_file_deps`` -- an ordinary cached call (no file reads, no
        output) keeps writing the same sparse two-key entry as before.
        Existing consumers that iterate backend metadata (``%cash_stats``,
        the explorer, eviction) already tolerate that sparse shape, and gain
        no new required field when these stay absent.
        """
        # `referenced`: the statement holding this result decides whether it
        # is worth its disk, and says so (`PersistencePolicy.decide`).
        metadata: dict[str, Any] = {"execution_time": elapsed, "timestamp": _time.time(), "referenced": True}
        if function:
            # What `cash inspect` names the entry by: a call key is `call:<sha>`,
            # so every intercepted call used to be listed as "call".
            metadata["function"] = function
        # `TieredBackend` reads exactly this key to bypass the ~0.1s
        # persistence floor, so threading the statement's resolved annotation
        # here is the whole fix -- the statement path writes the same field
        # from the same `force_persist` (`StatementStore.save`).
        #
        # Written only when True, keeping the sparse-entry shape every other
        # optional channel here follows, and it is a plain `bool`: metadata is
        # eagerly unpickled for EVERY entry at startup, so nothing but builtins
        # belongs in it (see the `callee_globals` note below).
        if self._persist_provider():
            metadata["force_persist"] = True
        if file_deps or remote_deps:
            try:
                # A file beside the callee's own code is checked in each
                # install's own copy, as the decorator records it.
                snap = attach_code_relative(snapshot_dependencies(file_deps, remote_deps), code_module)
            except Exception:  # noqa: BLE001 - never let dep snapshotting break the store
                snap = None
            if snap:
                metadata["auto_file_deps"] = snap
        # A statement holding this result stores a reference to this entry
        # (``call_refs``). Only for a call worth persisting -- hashing every
        # byte of a cheap call's result would cost more than the copy saves --
        # and not for one carrying captured globals, whose value is wrapped.
        #
        # Nor pickled whole when its size already refuses it: pickling
        # a 1.7 GiB result to learn that took 2.7 s after 2.8 s of
        # compute. It gets a one-off token for a digest and its estimated size
        # (`ESTIMATED_FIELD`): judged for disk on its own, and referred to only
        # by the statement it is the plain result of.
        #
        # Nor for the call a statement is nothing but (``a, b = build()``,
        # `plain_value`): that statement's reference is trusted without a
        # digest, so a token serves -- a 402 MiB result was worth
        # keeping, and pickling it for a digest took 2.6 s of 3.7.
        if elapsed >= _REF_MIN_COMPUTE_S and not callee_globals:
            estimate = self._too_big_to_digest(value, elapsed, plain_value)
            found = digest_and_size(value) if estimate is None else (UNHASHED_PREFIX + uuid.uuid4().hex, estimate)
            if found:
                metadata[DIGEST_FIELD], metadata[SIZE_FIELD] = found
                if estimate is not None:
                    metadata[ESTIMATED_FIELD] = True
                self._hold(key, value, *found)
        if stdout:
            metadata["stdout"] = stdout
        if stderr:
            metadata["stderr"] = stderr
        # Omitted when empty, like every other optional channel here,
        # so an ordinary cached call keeps writing the same sparse entry.
        #
        # The payload rides on the VALUE; metadata gets only a plain bool.
        # Metadata is unpickled for EVERY entry in the directory by anything
        # that surveys the cache -- `list_entries`, which `%cash_on` and the
        # CLI both call -- so a user object there is deserialised whether or
        # not it is ever used. Eviction used to do this too, on the write
        # worker, and no longer does: it ranks from a directory walk. A
        # survey of 5859 real metadata files found 31 of 33 fields are plain
        # builtins; the two that were not are a cash-owned serializer class and
        # a `numpy.int64` that leaked in as `size` and made those files
        # unreadable in any environment without numpy. This field must not be
        # the third.
        #
        # An earlier version put it in metadata to avoid changing the value
        # shape under an unchanged key. That objection is void: the `g:` key
        # component appears exactly when there are globals to capture, so an
        # entry carrying this payload has a key no earlier version could mint.
        # There is no old entry to collide with.
        if callee_globals:
            metadata["has_callee_globals"] = True
            value = (value, dict(callee_globals))
        try:
            self._cash.backend.set(key, value, metadata)
        except Exception:  # noqa: BLE001
            logger.debug("call unit: store failed for %s", key)

    @staticmethod
    def _too_big_to_digest(value, elapsed: float, plain_value: bool = False) -> int | None:
        """The estimated pickled size of *value* when it is not to be
        digested, else ``None``: when even half of it is more than its compute
        is worth on disk (half: a digest skipped for a value worth keeping
        costs other statements their reference), or when it is its
        statement's plain value, whose reference needs no digest."""

        estimate = pickled_size_estimate(value)
        if not estimate:
            return None  # nothing to estimate from: digest as before
        if not plain_value and worth_its_bytes(estimate // 2, elapsed):
            return None
        trace_event("call_digest_skipped", bytes_estimated=estimate, seconds=round(elapsed, 3))
        return estimate

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
