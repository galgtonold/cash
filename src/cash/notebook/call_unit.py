"""Runtime half of sub-statement caching: running one intercepted call.

``call_interception.py`` owns the AST half — which call nodes are structurally
eligible, and the rewrite. This module owns
:class:`CallCache`, which resolves a callee to its cached counterpart, the
runtime gate (:func:`call_site_is_cacheable`) and :class:`CallUnit`, which runs each call
through three collaborators: its key (:class:`~cash.notebook.call_key.CallKeys`,
which reads variable lineage), its entry (:class:`~cash.notebook.call_entries.CallEntries`,
the backend round-trip and the refusals before a write), and its effects
(:mod:`cash.notebook.call_effects`, recorded on a miss and replayed on a hit).

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
import itertools
import logging
import pathlib as _pathlib
import statistics
import sys
import time as _time
import types
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from cash._clock import perf_counter as _perf_counter
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability
from cash.analysis.handed_callables import hands_a_state_changing_callable
from cash.backends.persistence_policy import COMPUTE_FLOOR_S, restore_kind
from cash.cost_model import estimated_restore_time
from cash.notebook._trace import trace_event
from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_effects import (
    DigestHandoff,
    UNWRAP_FAILED,
    call_capturing_output,
    capture_globals,
    hash_args,
    reads_its_arguments_only,
    rebinds_its_closure,
    replay_deps,
    replay_output,
    restore_globals,
    unwrap_callee_globals,
)
from cash.notebook.call_entries import ONE_LOCAL, CallEntries, refs_beyond
from cash.notebook.call_interception import CallSite, SiteSlots, interceptable, names_read
from cash.notebook.call_key import CallKeys, callee_mutated_globals, global_digests
from cash.notebook.call_refs import (
    DIGEST_FIELD,
    SIZE_FIELD,
)
from cash.notebook.held_sentinels import find_for_call, put_back_for_call
from cash.notebook.pyplot_state import pyplot_state
from cash.sizing import estimate_object_size
from cash.tracking.file_tracker import FileAccessTracker, tracking_seconds
from cash.tracking.function_tracker import is_local_module
from cash.tracking.randomness import (
    capture_reachable_carrier_states,
    capture_rng_state,
    carrier_states_changed,
    rng_modules_changed,
)
from cash.value_types import INTERPRETER_MANAGED_GLOBALS

logger = logging.getLogger(__name__)

__all__ = ["call_site_is_cacheable", "CallCache", "CallUnit"]


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
#: ``for_handler`` uses to run a loop as one unit: once the calls of a site in
#: one statement run show that caching them costs more than
#: ``_OVERHEAD_FACTOR`` times what they compute -- after ``_GUARD_MIN_CALLS``
#: calls with no hit when that is plain, past ``_GUARD_AFTER_CALLS`` in any
#: case -- calls cheaper than ``_GUARD_CHEAP_BELOW_S`` are timed plain on a
#: few samples, and the site runs plain for the rest of the run when the
#: samples agree.
_GUARD_MIN_CALLS = 5
_GUARD_AFTER_CALLS = 50
_GUARD_CHEAP_BELOW_S = 0.05
_PLAIN_SAMPLES = 5
_OVERHEAD_FACTOR = 3.0
#: ... and when keying and looking a call up costs more than this share of the
#: call itself: the hit that follows also restores the value, so it would save
#: next to nothing.
_HIT_MUST_SAVE = 0.75


@functools.cache
def routed_call_s() -> float:
    """Seconds the rewritten line adds to a call its site runs plain
    (``call_interception._routed``: a compare, two subscripts and a
    counter), measured once on this interpreter.

    Left in the statement's time, it counted as the user's work: a
    comprehension of a million cheap calls computing in 0.07 s measured
    over 0.1 s, the floor past which its value is written to disk, and the
    cell waited 0.1 s for a write the plain kernel's work would not have
    earned. The least of a few runs, so a stall makes it smaller, never
    larger.
    """
    calls = 2000
    namespace = {"f": int, "P": [int], "C": [itertools.count(1).__next__], "r": range(calls)}
    routed = compile("for _ in r:\n    (f if f is P[0] and C[0]() else None)(0)", "<routed>", "exec")
    plain = compile("for _ in r:\n    f(0)", "<plain>", "exec")

    def best(code) -> float:
        times = []
        for _ in range(3):
            started = _perf_counter()
            exec(code, namespace)
            times.append(_perf_counter() - started)
        return min(times)

    try:
        return min(max(0.0, (best(routed) - best(plain)) / calls), 1e-6)
    except Exception:  # noqa: BLE001 - no measurement is no tax
        return 0.0


@dataclasses.dataclass
class _Invocation:
    """One cached call under way, and the cached calls made inside it."""

    #: What the calls made inside it that the cache holds stand for: their
    #: compute, and the predicted restore of their results.
    inner_compute_s: float = 0.0
    inner_restore_s: float = 0.0
    #: The same for this call itself, once it was served or stored.
    own: tuple[float, float] | None = None
    #: What this call would have cost to compute: its run time on a miss,
    #: its recorded cost on a hit, ``None`` when it ran plain. Kept per call:
    #: a call run plain inside its wrapper must not take on the outcome of
    #: a cached call made inside it.
    compute: float | None = None
    hit: bool = False
    #: Whether the call's result went to the cache (see ``_SiteRun.stored``).
    stored: bool = False
    #: Its ticket from `DigestHandoff.call_started`.
    ticket: tuple[int, int] = (0, 0)


@dataclasses.dataclass
class _SiteRun:
    """One call site's calls in the statement run under way."""

    calls: int = 0
    #: Per call through the cache, up to ``_GUARD_AFTER_CALLS`` of them: the
    #: seconds it took in all, what building its key and looking it up cost
    #: (all a hit pays, so a call cheaper than that is never worth caching),
    #: and what it computed (its run time on a miss, its recorded cost on a
    #: hit). Medians, never means: one stall does not decide.
    spent: list[float] = dataclasses.field(default_factory=list)
    keyed: list[float] = dataclasses.field(default_factory=list)
    computes: list[float] = dataclasses.field(default_factory=list)
    hits: int = 0
    #: How many of them were stored. A site whose calls were neither served
    #: nor stored -- each under the cost floor, or refused -- leaves nothing
    #: in the cache for a later call to be served from: caching it is all cost.
    stored: int = 0
    #: Whether the plain samples were taken after ``_GUARD_MIN_CALLS`` and
    #: found the site worth caching: it is judged again at
    #: ``_GUARD_AFTER_CALLS``.
    judged_early: bool = False
    probing: bool = False
    #: The calls run plain to time them, in seconds each.
    plain_samples: list[float] = dataclasses.field(default_factory=list)
    decided: bool = False
    plain: bool = False
    #: The median of the plain samples the site was decided on: what one of
    #: its calls is counted as taking once it runs plain.
    plain_s: float = 0.0


#: Where warnings re-emitted for a cash frame are de-duplicated, per file.
_WARNING_REGISTRIES: dict[str, dict] = {}
_CASH_DIR = _pathlib.Path(__file__).resolve().parents[1]


def _in_cash(filename: str) -> bool:
    try:
        return _pathlib.Path(filename).resolve().is_relative_to(_CASH_DIR)
    except (OSError, ValueError, TypeError):
        return False


_filters_checked: tuple = ()
_filters_error = False


def _errors_on_every_warning() -> bool:
    """Whether a filter turns every warning into an error.

    Asked on every intercepted call, so the answer is kept for as long as the
    filter list holds the same entries (a tuple comparison that stops at the
    first entry that is a different object, instead of a scan of the list).
    """
    global _filters_checked, _filters_error
    current = tuple(warnings.filters)
    if current == _filters_checked:
        return _filters_error
    _filters_error = any(
        action == "error" and category is Warning and message is None and module is None
        for action, message, category, module, _lineno in current
    )
    _filters_checked = current
    return _filters_error


class _warnings_at_the_caller:
    """Re-emit warnings raised inside an intercepted call, at the user's line.

    ``warnings.warn(..., stacklevel=2)`` names the frame that called the
    function -- which, under interception, is cash's wrapper. A pandas warning
    quoted ``result = fn(*args, **kwargs)`` from ``call_unit.py`` where the
    user's own line belonged. Recorded, then re-emitted with
    a cash frame replaced by the first user frame above it, de-duplicated per
    location as the default filter would. A filter that turns warnings into
    errors is left to act as it would: recording would swallow the exception.

    A class, not a generator: the plain-run calls of a loop pay this on every
    call.
    """

    __slots__ = ("_caught", "_catcher")

    def __enter__(self):
        if _errors_on_every_warning():
            self._catcher = None
            return None
        self._catcher = warnings.catch_warnings(record=True)
        self._caught = self._catcher.__enter__()
        warnings.simplefilter("always")
        return None

    def __exit__(self, *_exc):
        catcher = self._catcher
        if catcher is None:
            return False
        catcher.__exit__(None, None, None)
        caught = self._caught
        if caught:
            _relay_warnings(caught)
        return False


def _relay_warnings(caught) -> None:
    """Re-emit what an intercepted call warned, with cash's frames replaced."""
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
        except Exception:  # relaying a warning never breaks the call
            logger.debug("call unit: could not relay a warning", exc_info=True)
            try:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            except Exception:  # noqa: BLE001 - relaying a warning never breaks the call
                pass


@dataclasses.dataclass(frozen=True)
class _Call:
    """One keyed call on its way through :meth:`CallUnit.wrap`."""

    fn: Callable[..., Any]
    site: CallSite
    func_name: str
    key: str
    #: The globals the callee writes (``callee_mutated_globals``).
    mutated_globals: tuple[str, ...]
    args: tuple
    kwargs: dict


def _decide_site(run: _SiteRun, site: CallSite) -> None:
    """End *run*'s probe: from the timed plain samples and the cached calls
    before them, decide whether the rest of the site's calls run plain."""
    run.probing = False
    # Medians, not means: one sample stretched by a stall (the process
    # descheduled, a garbage collection) lifted the mean of five plain
    # samples over the bar, and the site stayed cached for the rest of the
    # run; on the cached side the first call also builds what later keys
    # reuse.
    cached = statistics.median(run.spent)
    plain = run.plain_s = statistics.median(run.plain_samples)
    keyed = statistics.median(run.keyed)
    # Too dear to cache, or a hit could not save a quarter of
    # the call: a hit pays the key and lookup, then the restore.
    run.plain = (
        cached > (1 + _OVERHEAD_FACTOR) * plain or keyed >= _HIT_MUST_SAVE * plain or _nothing_to_serve(run)
    )
    early = run.calls < _GUARD_AFTER_CALLS
    run.decided = run.plain or not early
    if not run.decided:
        # Worth caching on the first few calls: judged again, on fresh
        # samples, once the site has made ``_GUARD_AFTER_CALLS`` calls.
        run.judged_early = True
        run.plain_samples.clear()
    trace_event(
        "call_site_decided",
        source=site.source,
        calls=run.calls,
        cached_ms=round(cached * 1000, 3),
        keyed_ms=round(keyed * 1000, 3),
        plain_ms=round(plain * 1000, 3),
        plain=run.plain,
    )


def _nothing_to_serve(run: _SiteRun) -> bool:
    """Whether none of *run*'s calls was served or stored: under the cost
    floor (``call_cost_floor_s``) a miss is not kept, so a site of 1 ms calls
    paid ~0.4 ms a call for a key and a lookup that could never be served,
    +0.5 s over 1,000 calls."""
    return not run.hits and not run.stored


def _time_to_probe(run: _SiteRun) -> bool:
    """Whether *run*'s calls have shown enough to time a few plain.

    Past ``_GUARD_AFTER_CALLS`` calls, whenever they are cheap. Before that,
    from ``_GUARD_MIN_CALLS`` on, only when the evidence is already plain:
    no hit, and either nothing stored (`_nothing_to_serve`) or caching a
    call costs more than the verdict's bar with the call's own time under
    the cache -- never less than its plain time -- standing in for the plain
    time. A short comprehension of a cheap helper pays a handful of keys
    instead of fifty."""
    if not run.computes:
        return False
    compute = statistics.median(run.computes)
    if compute >= _GUARD_CHEAP_BELOW_S:
        return False
    if run.calls >= _GUARD_AFTER_CALLS:
        return True
    if run.judged_early or run.hits or run.calls < _GUARD_MIN_CALLS:
        return False
    return (
        _nothing_to_serve(run)
        or statistics.median(run.spent) > (1 + _OVERHEAD_FACTOR) * compute
        or statistics.median(run.keyed) >= _HIT_MUST_SAVE * compute
    )


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
        digests: DigestHandoff | None = None,
    ):
        self._cash = cash_instance
        #: The statement's argument digests, shared with its in-place-change
        #: check (`DigestHandoff`).
        self._digests = digests if digests is not None else DigestHandoff()
        #: Builds each call's key and remembers what each site was keyed on.
        self._keys = CallKeys(ctx_provider, loop_vars_provider, loop_var_digests_provider)
        #: Looks call entries up and writes them; holds the sites refused.
        self._entries = CallEntries(cash_instance, ttl_provider, persist_provider)
        self.call_log: list[dict] = []
        #: The record each plain-run (site, callee) is counted in.
        self._plain_records: dict[tuple[str, str, int], dict] = {}
        #: Per call site, how its calls went in the statement run under way
        #: (see :meth:`_entry_for`); emptied by :meth:`begin_statement`.
        self._site_runs: dict[CallSite, _SiteRun] = {}
        #: The sites the guard switched to run plain in the statement run under
        #: way: ``site -> [fn, its pooled record, seconds a call takes, calls
        #: not yet in the record]``. :meth:`CallCache.resolve` hands such a
        #: site's *fn* back as it is and counts the call in the last slot;
        #: :meth:`_count_plain_calls` moves the count into the record, at the
        #: time the timed calls took, before anything reads or drops it.
        self.plain_sites: dict[CallSite, list] = {}
        #: Seconds this unit spent on calls beyond their own compute (keys,
        #: lookups, stores, restores), and the compute its hits stood in for.
        #: Monotonic; a statement reads the difference across its run (see
        #: ``CallRouting.price``). The overhead is that of calls made outside
        #: every other call's run: one made inside a call that runs is part
        #: of that call's measured time, which is what its entry records.
        self.overhead_s = 0.0
        self.hits_saved_s = 0.0
        #: The run time of the calls that ran through the cache, outside every
        #: other call's run, and the seconds the file tracker spent inside
        #: those calls and the ones served. The tracker's seconds inside a
        #: call are already in ``overhead_s`` or in the call's own time: a
        #: statement does not take them off again. Monotonic.
        self.computed_s = 0.0
        self.tracking_in_calls_s = 0.0
        #: How many calls are running their function right now (see
        #: :meth:`_run_miss`); a call made while one is, is part of its time.
        self._running = 0
        #: The compute of the calls the cache now holds -- a stored miss's
        #: run time, a hit's recorded cost -- and the predicted restore time
        #: of their results. What a statement's own work leaves out (see
        #: ``CallRouting.price``). Outermost calls only: a cached call's
        #: compute already holds the calls made inside it. Monotonic, like
        #: those above.
        self.cached_compute_s = 0.0
        self.cached_restore_s = 0.0
        self._invocations: list[_Invocation] = []
        #: The files and URLs read inside calls the cache holds, each with the
        #: sequence number of its last such read (see :meth:`files_read_since`).
        self._cached_reads: dict[str, int] = {}
        self.reads_seq = 0
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
        #: The hashes of the frames this cell's calls received, checked
        #: rather than read again while unchanged; emptied by :meth:`begin_cell`.
        self._fingerprints = self._digests.fingerprints

    def _cost_floor_s(self) -> float:
        """The bar a call's own execution must clear to be stored (:func:`call_cost_floor_s`)."""
        return call_cost_floor_s(self._cash)

    def begin_cell(self) -> None:
        """A new cell: the results held for references, and the frames
        remembered for their hashes, are let go."""
        self.held_results.clear()
        self._fingerprints.clear()

    def _hold(self, key: str, value: Any, digest: str | None, size: Any) -> None:
        if digest:
            self.held_results[id(value)] = (value, key, digest, size if isinstance(size, int) else 0)

    def begin_statement(self) -> None:
        """A new statement run: every site starts over (see :meth:`_entry_for`)."""
        self._count_plain_calls()
        self._site_runs.clear()
        self.plain_sites.clear()
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
        here, by measurement, with the loop's own numbers: once the calls
        show caching them costs more than ``_OVERHEAD_FACTOR`` times what they
        compute (``_time_to_probe``: from ``_GUARD_MIN_CALLS`` calls with no
        hit when that is plain, past ``_GUARD_AFTER_CALLS`` in any case), if
        they are cheap, a few are run plain and timed, and when the samples
        agree the rest of the statement's calls to this site run plain.
        Running plain is always correct; it is only uncached.
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
            __tracebackhide__ = True
            with _warnings_at_the_caller():
                return _guarded(*args, **kwargs)

        def _guarded(*args, **kwargs):
            # IPython leaves a frame with this set out of the traceback it
            # prints: a user's KeyError showed three of cash's wrapper frames
            # between their cell and their function.
            __tracebackhide__ = True
            ticket = self._digests.call_started()
            run = self._site_runs.get(site)
            if run is None:
                run = self._site_runs[site] = _SiteRun()
            if run.plain:
                started = _perf_counter()
                result = fn(*args, **kwargs)
                _log_plain(_perf_counter() - started)
                if site not in self.plain_sites:
                    # Decided at the re-check, whose calls are not logged as
                    # run plain: the record exists from this call on.
                    record = self._plain_records.get((names[0], site.source, site.occurrence_index))
                    if record is not None:
                        self.plain_sites[site] = [fn, record, run.plain_s, 0]
                self.last_returned = (None, id(result), site.source)
                return result
            if run.probing and run.calls < _GUARD_AFTER_CALLS:
                started = _perf_counter()
                result = fn(*args, **kwargs)
                took = _perf_counter() - started
                run.plain_samples.append(took)
                _log_plain(took)
                if len(run.plain_samples) >= _PLAIN_SAMPLES:
                    _decide_site(run, site)
                    record = self._plain_records.get((names[0], site.source, site.occurrence_index)) if names else None
                    if run.plain and record is not None:
                        self.plain_sites[site] = [fn, record, run.plain_s, 0]
                self.last_returned = (None, id(result), site.source)
                return result
            self._last_key_s = None
            # One slot per invocation: a call the callee makes through a
            # lambda it was handed runs its own `_invoke` inside this one.
            self._invoked_keys.append(None)
            invocation = _Invocation(ticket=ticket)
            self._invocations.append(invocation)
            outside = self._running == 0
            tracked = tracking_seconds() if outside else 0.0
            started = _perf_counter()
            try:
                result = invoke(*args, **kwargs)
            finally:
                invoked_key = self._invoked_keys.pop()
                self._count_cached(self._invocations.pop())
            spent = _perf_counter() - started
            self.last_returned = (invoked_key, id(result), site.source)
            compute = invocation.compute
            if compute is not None:
                self._add_to_clocks(invocation, spent, tracking_seconds() - tracked if outside else None)
            run.calls += 1
            if run.decided:
                return result
            if run.probing:
                # The re-check past `_GUARD_AFTER_CALLS`: its calls still go
                # through the cache, and what each would cost to compute (its
                # run time, or a hit's recorded cost) stands in for its plain
                # time. Run plain, they were neither looked up nor stored,
                # whatever they cost: in `[work(i) for i in range(60)]` the
                # 51st to 55th calls ran again on every re-run, 0.6 s each.
                run.plain_samples.append(compute if compute is not None else spent)
                if invocation.hit:
                    run.hits += 1
                if invocation.stored:
                    run.stored += 1
                if len(run.plain_samples) >= _PLAIN_SAMPLES:
                    _decide_site(run, site)
                return result
            if len(run.spent) < _GUARD_AFTER_CALLS:
                run.spent.append(spent)
                run.keyed.append(self._last_key_s or 0.0)
                if compute is not None:
                    run.computes.append(compute)
            if invocation.hit:
                run.hits += 1
            if invocation.stored:
                run.stored += 1
            if _time_to_probe(run):
                run.probing = True
            return result

        return _entry

    def _add_to_clocks(self, invocation: _Invocation, spent: float, tracked: float | None) -> None:
        """Add a call that was served or ran through the cache to the
        statement's clocks: *spent* seconds in all, *tracked* of them (or
        ``None`` when it was made inside another call's run) recording reads.

        A hit's compute is what it stood in for, however deep. The rest is
        counted only for a call made outside every other call's run: one
        made inside is part of that call's measured time."""
        compute = invocation.compute or 0.0
        if invocation.hit:
            self.hits_saved_s += compute
        if tracked is None:
            return
        self.tracking_in_calls_s += max(0.0, tracked)
        if invocation.hit:
            self.overhead_s += spent
        else:
            self.overhead_s += max(0.0, spent - compute)
            self.computed_s += compute

    def _outcome(self, compute: float, *, hit: bool) -> None:
        """What the call under way would have cost to compute (see ``_Invocation.compute``)."""
        if self._invocations:
            self._invocations[-1].compute = compute
            self._invocations[-1].hit = hit

    def _count_cached(self, invocation: _Invocation) -> None:
        """Add what *invocation* leaves in the cache to the call around it, or
        to the totals when it is outermost: the call itself when it was served
        or stored, else the cached calls made inside it."""
        compute, restore = invocation.own or (invocation.inner_compute_s, invocation.inner_restore_s)
        if self._invocations:
            outer = self._invocations[-1]
            outer.inner_compute_s += compute
            outer.inner_restore_s += restore
        else:
            self.cached_compute_s += compute
            self.cached_restore_s += restore

    def _cached(self, compute: float, value: Any, reads: frozenset[str]) -> None:
        """The call under way is now held by the cache: it stands for
        *compute* seconds, returned *value*, and read *reads*.

        Only a call its entry keeps past a restart counts: one that took the
        persistence floor (``COMPUTE_FLOOR_S``), or that ``persist`` sends to
        disk. A quicker one lives in RAM only, so after a restart the
        statement around it is what would bring its result back."""
        if compute < COMPUTE_FLOOR_S and not self._entries.persists():
            return
        if self._invocations:
            self._invocations[-1].own = (compute, self._restore_estimate(value))
        if reads:
            self.reads_seq += 1
            for path in reads:
                self._cached_reads[path] = self.reads_seq

    def _restore_estimate(self, value: Any) -> float:
        """Seconds the cost model predicts restoring *value* takes, from the
        tier a statement's value is restored from."""
        try:
            kind = restore_kind(getattr(self._cash, "backend", None))
            return estimated_restore_time(type(value).__name__, estimate_object_size(value), kind)
        except Exception:  # noqa: BLE001 - an estimate is never worth an error
            return 0.0

    def files_read_since(self, seq: int) -> frozenset[str]:
        """Files and URLs read inside cached calls since :attr:`reads_seq` was *seq*."""
        return frozenset(path for path, at in self._cached_reads.items() if at > seq)

    def wrap(self, fn, site: CallSite):
        func_name = self._func_name(fn)

        def _invoke(*args, **kwargs):
            __tracebackhide__ = True
            if rebinds_its_closure(fn):
                # `counter()` bumps a `nonlocal`: a hit would skip the bump
                # and hand the next call the same count.
                return fn(*args, **kwargs)
            if hands_a_state_changing_callable(fn, args, kwargs):
                # `run_steps(steps, data)` with `steps = [f]`, `f` appending
                # to a list: the callee runs `f`, and a hit would skip it.
                return fn(*args, **kwargs)
            # Globals this callee writes. Resolved per call rather
            # than once per `wrap`, because the underlying source analysis is
            # memoised (`callee_mutated_globals`) while the "is it bound, is it
            # a module" filter genuinely depends on the live namespace. Empty
            # for nearly every callee, and every step below short-circuits on
            # empty, so an ordinary call pays one memo lookup.
            key_started = _perf_counter()
            mutated_globals = callee_mutated_globals(fn)
            key = self._keys.key(
                site,
                args,
                kwargs,
                global_digests(fn, mutated_globals) if mutated_globals else None,
                # A callee that writes globals keys on their state, not on
                # what it receives.
                fn=None if mutated_globals else fn,
            )
            if key is None:
                # Either the key build raised, or `call_cache_key` itself
                # refused (arg-digest count mismatch, by design). Either way
                # this call is not safely keyable right now -- run it
                # uncached rather than risk a collapsed, wrong key.
                return self._run_unkeyed(fn, args, kwargs, key_started)

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
                return self._run_unkeyed(fn, args, kwargs, key_started)

            call = _Call(fn, site, func_name, key, mutated_globals, args, kwargs)
            hit_started = _perf_counter()
            hit, value, recorded_cost, metadata = self._entries.lookup(key)
            self._last_key_s = _perf_counter() - key_started
            if hit:
                return self._serve_hit(call, value, recorded_cost, metadata, hit_started)
            return self._run_miss(call)

        return self._entry_for(fn, site, _invoke)

    def _run_unkeyed(self, fn, args: tuple, kwargs: dict, key_started: float):
        """Run a call plain that the cache will not serve, and say what it and
        its key cost, so a site whose calls are all like this is judged by the
        guard (``_decide_site``) instead of paying for a key on every call."""
        __tracebackhide__ = True
        self._last_key_s = _perf_counter() - key_started
        started = _perf_counter()
        result = fn(*args, **kwargs)
        self._outcome(_perf_counter() - started, hit=False)
        return result

    def _serve_hit(self, call: _Call, value, recorded_cost: float, metadata: Mapping[str, Any], hit_started: float):
        """Hand back a found entry's value, with the call's effects put back."""
        __tracebackhide__ = True
        value, captured_globals = unwrap_callee_globals(value, metadata)
        if value is UNWRAP_FAILED:
            # The entry says it carries captured globals and does not.
            # Treat it as absent rather than hand back a tuple where a
            # value belongs -- a miss costs a recompute, this would be
            # a silently wrong value.
            return call.fn(*call.args, **call.kwargs)
        # Replay the ORIGINAL execution's observations into the
        # statement's ambient capture (FileAccessTracker, live
        # stdout/stderr). The call itself does not run on a hit, so
        # without this replay the tracker records no read and the
        # live stream sees no print -- the enclosing statement's own
        # entry would then be rewritten (on ITS next miss) from a
        # degraded observation that is missing both. Mirrors
        # ``propagate_file_deps_to_active_tracker`` in decorator/file_deps.py,
        # the ``@cash.cache`` decorator's defence against the same
        # failure mode.
        reads = replay_deps(metadata)
        replay_output(metadata)
        restore_globals(call.fn, call.mutated_globals, captured_globals)
        value = put_back_for_call(call.fn, value, metadata.get("held_refs"))
        if not captured_globals:
            self._hold(call.key, value, metadata.get(DIGEST_FIELD), metadata.get(SIZE_FIELD))
        self._record(call.func_name, call.site, call.key, cache_hit=True, elapsed=0.0, time_saved=recorded_cost)
        self._outcome(recorded_cost or 0.0, hit=True)
        self._cached(recorded_cost or 0.0, value, reads)
        self._entries.drop_if_hit_costs_more(call.key, _perf_counter() - hit_started, recorded_cost)
        return value

    def _run_miss(self, call: _Call):
        """Run the call, watching what it does, and keep its result when a
        later hit could stand in for it."""
        __tracebackhide__ = True
        # The call runs inside the STATEMENT's ambient capture
        # (FileAccessTracker, RNG capture, output capture), which wraps
        # the whole statement's exec. On a genuine miss that capture
        # records this call's effects as its own, for free, and
        # everything is correct. The broken case is a later run where the
        # STATEMENT misses and re-executes but the CALL hits: the call's
        # effects do not re-happen, so the statement's own capture is
        # rewritten from a degraded observation. The checks in
        # :meth:`_did_what_a_hit_cannot` fail CLOSED -- refuse the site
        # outright rather than serve a value whose side effects will not
        # re-happen.
        #
        # File deps and stdout/stderr are recorded around the call so a
        # LATER hit can replay them (:meth:`_serve_hit`). RNG is not
        # recorded -- a call that consumed it is refused, whether it drew from
        # a module's stream or from a generator it reaches through its globals
        # (``rng`` built in a cell, drawn from inside the helper).
        #
        # A NESTED tracker, not a before/after diff against the ambient
        # (statement-wide) one, as the decorator wraps its call in a fresh
        # ``FileAccessTracker(propagate_to_parent=True)``. That distinction
        # is load-bearing: ``active_tracker.get()`` is shared for the whole
        # statement (or, inside a loop, the whole loop-as-one-unit
        # execution), so a diff against it goes silently wrong the moment the
        # SAME path is read twice in one tracker window -- ``hdr = read(p);
        # total = expensive(k)``, or two loop iterations both reading the
        # same file. The second read's "after" set already contains the path
        # from the first, so ``after - before`` is EMPTY: the entry stores no
        # dependency at all, and ``CallEntries._auto_file_deps_fresh`` is
        # vacuously true forever. A fresh, per-call tracker has no such
        # baseline to collide with -- its own set IS this call's reads, full
        # stop -- and ``propagate_to_parent=True`` still surfaces every read
        # to the enclosing statement's tracker immediately, so the miss-path
        # "recorded for free" behaviour holds.
        rng_before = capture_rng_state(), capture_reachable_carrier_states(call.fn)
        figures_before = pyplot_state()
        ticket = self._invocations[-1].ticket if self._invocations else (0, 0)
        arg_hashes_before = (
            None
            if reads_its_arguments_only(call.fn)
            else hash_args(call.args, call.kwargs, digest=self._digests.digest_before(ticket))
        )
        module_before = _module_globals(call.fn)
        started = _perf_counter()
        call_tracker = FileAccessTracker(
            getattr(call.fn, "__globals__", None),
            propagate_to_parent=True,
        )
        self._running += 1
        try:
            with call_tracker:
                result, stdout_text, stderr_text = call_capturing_output(call.fn, call.args, call.kwargs)
        finally:
            self._running -= 1
        elapsed = _perf_counter() - started
        # Read while `result` is this frame's only reference of its own.
        result_held = refs_beyond([result]) > ONE_LOCAL
        stored = False
        if self._did_what_a_hit_cannot(call, rng_before, arg_hashes_before, ticket) or _rebound_unwatched(
            call, module_before
        ):
            self._entries.refuse(call.key)
        # Not stored when it made, picked or drew on a pyplot figure, which a
        # hit would not: the next drawing would land on another figure.
        elif pyplot_state() == figures_before and self._worth_storing(call, result, elapsed, result_held):
            stored = self._store_result(call, result, elapsed, call_tracker, stdout_text, stderr_text)
        if stored:
            self._cached(
                elapsed,
                result,
                frozenset(call_tracker.get_accessed_files()) | frozenset(call_tracker.get_accessed_remote_urls()),
            )
        self._record(call.func_name, call.site, call.key, cache_hit=False, elapsed=elapsed, stored=stored)
        self._outcome(elapsed, hit=False)
        if stored and self._invocations:
            self._invocations[-1].stored = True
        return result

    def _did_what_a_hit_cannot(
        self, call: _Call, rng_before, arg_hashes_before: tuple | None, ticket: tuple[int, int] = (0, 0)
    ) -> bool:
        """Whether the call just run had an effect a hit would silently skip.

        *arg_hashes_before* is None for a callee that never writes into its
        arguments (`reads_its_arguments_only`): they are not hashed again."""
        modules_before, carriers_before = rng_before
        if rng_modules_changed(modules_before, capture_rng_state()) or carrier_states_changed(carriers_before):
            # RNG is a consumed linear resource -- what matters is stream
            # POSITION, not membership. A hit leaves the global stream
            # where it was, so every downstream draw would diverge from
            # the uncached oracle. Replaying it properly needs a
            # sub-statement position anchor that does not exist, so the
            # site is refused instead.
            return True
        # The callee mutated a live argument in place and returned
        # something else (`df.dropna(inplace=True); return len(df)`).
        # The identity check only catches `return arg` -- this
        # catches "mutated but returned a *different* object", which
        # a hit would silently skip.
        if arg_hashes_before is None:
            return False
        arg_hashes_after = hash_args(call.args, call.kwargs, self._fingerprints)
        if arg_hashes_after != arg_hashes_before:
            return True
        # Unchanged: what was just read is what the statement's own check
        # after it would read (`DigestHandoff.note_after`).
        self._digests.note_after(ticket, (*call.args, *call.kwargs.values()), arg_hashes_after)
        return False

    def _worth_storing(self, call: _Call, result, elapsed: float, result_held: bool = True) -> bool:
        """Past the cost floor, safe to hand back as a copy, and cheaper to
        restore than to compute again."""
        return (
            elapsed >= self._cost_floor_s()
            and self._entries.storable(result, call.args, call.kwargs, call.fn, result_held)
            and self._entries.restore_pays(result, elapsed)
        )

    def _store_result(
        self, call: _Call, result, elapsed: float, call_tracker: FileAccessTracker, stdout_text: str, stderr_text: str
    ) -> bool:
        """Write the result with what the call read, printed and wrote to its
        globals; ``False`` when those globals cannot be captured."""
        # The callee's writes to its own globals, captured as
        # an END STATE. Snapshotting the final value needs no ordering
        # and no idempotence, which is why this is tractable where
        # replaying the individual mutations is not.
        #
        # `None` means "cannot capture this soundly" -- an unpicklable
        # value, or one whose hash falls back to identity so a later
        # pre-state comparison could not tell it had changed. Refuse
        # the SITE rather than store an entry whose restore would be
        # wrong, matching the RNG and argument-mutation refusals:
        # an uncached call is merely slow.
        captured = capture_globals(call.fn, call.mutated_globals)
        if captured is None:
            self._entries.refuse(call.key)
            return False
        held = self._entries.store(
            call.key,
            result,
            elapsed,
            file_deps=frozenset(call_tracker.get_accessed_files()),
            remote_deps=frozenset(call_tracker.get_accessed_remote_urls()),
            stdout=stdout_text,
            stderr=stderr_text,
            callee_globals=captured,
            function=call.func_name,
            plain_value=call.site.source == self.plain_value_source,
            code_module=getattr(call.fn, "__module__", None),
            held_refs=find_for_call(call.fn, result),
        )
        if held:
            self._hold(call.key, result, *held)
        return True

    def _func_name(self, fn) -> str:
        """The name this call's events display under in the badge and stats.

        Delegates to ``Cash.get_func_key`` for a stable ``module.qualname``
        rather than rebuilding the rule here.
        """
        try:
            return self._cash.get_func_key(fn)
        except Exception:  # noqa: BLE001 - a display name must never break the call
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
        if ran_plain:
            # One record per site and callee, counting its calls: a loop
            # over 85,000 lines made 2 million records for the badge to draw.
            pooled_key = (func_name, site.source, site.occurrence_index)
            pooled = self._plain_records.get(pooled_key)
            if pooled is not None:
                pooled["calls"] += 1
                pooled["execution_time"] += elapsed
                return
        record = {
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
            # How many calls this record stands for (see above).
            "calls": 1,
        }
        if ran_plain:
            self._plain_records[(func_name, site.source, site.occurrence_index)] = record
        self.call_log.append(record)

    def _count_plain_calls(self) -> None:
        """Move the calls :meth:`CallCache.resolve` counted for the plain
        sites into their records (see `plain_sites`)."""
        for entry in self.plain_sites.values():
            calls, entry[3] = entry[3], 0
            if calls:
                record = entry[1]
                record["calls"] += calls
                record["execution_time"] += calls * entry[2]
                # Cash's, not the user's: the statement leaves it out of
                # what it cost (``statement_price``).
                self.overhead_s += calls * routed_call_s()

    def drain(self) -> list[dict]:
        self._count_plain_calls()
        events, self.call_log = self.call_log, []
        self._plain_records = {}
        # Their records left with the events: the next calls log new ones.
        self.plain_sites.clear()
        return events


#: What a site's `CallCache.plain_callees` slot holds while it does not run
#: plain: no callee is this object.
_NOT_PLAIN = object()


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
        *,
        ctx_provider: Callable[[], CacheKeyContext],
        loop_vars_provider: Callable[[], dict[str, Any]],
        loop_var_digests_provider: Callable[[], dict[str, str]],
        ttl_provider: Callable[[], int | None],
        persist_provider: Callable[[], bool],
        slots: SiteSlots | None = None,
        digests: DigestHandoff | None = None,
    ):
        self._cash = cash_instance
        # Keyed by (id(fn), site) -- NOT (id(fn), site_index). `set_sites` is
        # called once per STATEMENT, so `site_index` (an index into that
        # statement's own site list) is reused across every statement that
        # rewrites at least one call: index 0 means something different on
        # every statement. Keying on the index let editing a cell reuse the
        # PREVIOUS statement's wrapper -- same fn, index 0 -- serving its
        # source/free_names/computed_arg_positions after the callee's own
        # argument expression had changed (reproduced as
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
        # `stmt_identity` joining this tuple is deliberate, not
        # incidental: two statements that previously built an EQUAL CallSite
        # (same call text, same free names, same occurrence index) now build
        # DIFFERENT ones, so each statement gets its own wrapper instead of
        # silently sharing one minted for the other. That re-scoping is
        # exactly what fixes the underlying key collision -- a shared wrapper
        # closes over one `site`, and a wrapper reused across statements would
        # still build the collapsed key `stmt_identity` exists to prevent.
        self._wrappers: dict[tuple[int, CallSite], tuple[types.FunctionType, object]] = {}
        # NOTE: there is deliberately no name-reconciliation here any more.
        # This class used to rebuild ``module.qualname`` via
        # ``Cash.get_func_key`` so the badge could tell an intercepted call
        # from a hand-decorated one, with a comment warning that the two "must
        # agree exactly or the badge silently stops marking intercepted calls".
        # Call-unit events set ``intercepted=True`` at the source, so the two
        # can no longer drift.
        #: The numbers the rewritten lines name their sites by (see
        #: ``call_interception.SiteSlots``), and the sites by number. A lambda
        #: or a generator expression runs its line in whatever statement calls
        #: it: looked up in the statement's own table, its call was keyed as
        #: that statement's site of the same index.
        self._slots = slots if slots is not None else SiteSlots()
        self._site_list: list[CallSite] = self._slots.sites
        #: The number of the first site :meth:`set_sites` was handed last, for
        #: a :meth:`resolve` that names none.
        self._first_slot = 0
        self._call_unit = CallUnit(
            cash_instance,
            ctx_provider,
            loop_vars_provider,
            loop_var_digests_provider,
            ttl_provider,
            persist_provider,
            digests,
        )
        #: The unit's sites run plain (one dict, emptied in place), read on
        #: every call.
        self._plain_sites = self._call_unit.plain_sites
        #: The same by the index of this statement's site, emptied with it.
        self._plain_at: dict[int, list] = {}
        #: Bound in ``user_ns`` (``call_interception.PLAIN_NAME`` and
        #: ``COUNT_NAME``), indexed by site number: the callee each site runs
        #: plain, and a counter of the calls the rewritten line made to it
        #: without calling `resolve`. Only ever grown, to every number given
        #: out, so a number a lambda kept from an earlier statement still
        #: reads a slot.
        self.plain_callees: list = []
        self.plain_counters: list = []
        #: Site index -> its `CallUnit.plain_sites` entry, for the sites whose
        #: slots are set: their counts are moved there by `_fold_counters`.
        self._counted_at: dict[int, list] = {}

    def begin_cell(self) -> None:
        self._call_unit.begin_cell()

    @property
    def call_unit(self) -> CallUnit:
        """The unit that keys, stores and serves this cache's calls."""
        return self._call_unit

    def held_results(self) -> dict:
        return self._call_unit.held_results

    def outermost_result(self):
        return self._call_unit.outermost_result()

    def _run_direct(self, site_index: int, plain: list) -> None:
        """Let the rewritten line call *plain*'s callee itself, counting each
        call (see ``call_interception._routed``)."""
        if site_index < len(self.plain_callees) and site_index not in self._counted_at:
            self.plain_counters[site_index] = itertools.count(1).__next__
            self.plain_callees[site_index] = plain[0]
            self._counted_at[site_index] = plain

    def _fold_counters(self) -> None:
        """Move the calls the rewritten lines counted into their sites'
        entries, and close the slots: the next call goes through `resolve`."""
        for site_index, plain in self._counted_at.items():
            self.plain_callees[site_index] = _NOT_PLAIN
            # The counter starts at 1: its next value is the calls plus one.
            plain[3] += self.plain_counters[site_index]() - 1
            self.plain_counters[site_index] = None
        self._counted_at.clear()

    def slot_for(self, site: CallSite) -> int:
        """The number the rewritten line names *site* by (``SiteSlots.slot_for``)."""
        return self._slots.slot_for(site)

    def set_sites(self, sites: list[CallSite], plain_value_source: str | None = None) -> None:
        """A statement whose calls are rewritten is about to run: *sites* are
        numbered (if the rewrite has not already), and each site's guard
        starts over."""
        self._fold_counters()
        slot_for = self._slots.slot_for
        slots = [slot_for(site) for site in sites]
        self._first_slot = slots[0] if slots else 0
        grow = len(self._site_list) - len(self.plain_callees)
        if grow > 0:
            self.plain_callees.extend([_NOT_PLAIN] * grow)
            self.plain_counters.extend([None] * grow)
        self._plain_at.clear()
        # One call per statement run: each site's guard starts over.
        self._call_unit.begin_statement()
        self._call_unit.plain_value_source = plain_value_source

    def fold_plain_counts(self) -> None:
        """Count the calls the rewritten lines made to the sites run plain
        into their records and the unit's overhead (see `_fold_counters`)."""
        self._fold_counters()
        self._call_unit._count_plain_calls()

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
        self._fold_counters()
        self._plain_at.clear()  # the unit empties `plain_sites`
        return self._call_unit.drain()

    def resolve(self, fn, site_index: int | None = None):
        """Return *fn* or a cached counterpart. Never raises.

        *site_index* is the site's number (``SiteSlots``); none names the
        first site of the statement :meth:`set_sites` was handed last."""
        if site_index is None:
            site_index = self._first_slot
        plain = self._plain_at.get(site_index)
        if plain is not None and plain[0] is fn:
            # The guard runs the rest of this site's calls plain: the
            # function itself, called from the user's own line, with no
            # wrapper to enter and no warnings to relay. A comprehension
            # calling a cheap helper 200,000 times paid ~8 us a call for the
            # wrapper -- 60x the plain kernel. Asked first, and counted in
            # one slot (`CallUnit.plain_sites`): this is the whole cost such
            # a call pays.
            plain[3] += 1
            return fn
        try:
            site = self._site_list[site_index]
        except (IndexError, TypeError):
            # No site under this number (a number from another numbering):
            # a call with no site has nothing to be keyed on, so it runs plain.
            return fn

        if self._plain_sites:
            # Looked up by the site, which compares field by field (the
            # guard's site is an equal object, not this one): once.
            plain = self._plain_sites.get(site)
            if plain is not None:
                self._plain_at[site_index] = plain
                if plain[0] is fn:
                    plain[3] += 1
                    self._run_direct(site_index, plain)
                    return fn
        if not interceptable(fn):
            return fn

        # Keyed on the SITE, not the index -- see the long comment on
        # `_wrappers` in `__init__` for why the index alone is unsafe.
        cache_key = (id(fn), site)
        entry = self._wrappers.get(cache_key)
        if entry is not None and entry[0] is fn:
            return entry[1]

        try:
            wrapper = self._call_unit.wrap(fn, site)
        except Exception:  # noqa: BLE001 - a caching wrapper is never worth an error
            return fn
        self._wrappers[cache_key] = (fn, wrapper)
        return wrapper


def _module_globals(fn) -> dict[str, Any] | None:
    """What the globals of the local module *fn* is defined in hold before
    it runs, for `_rebound_unwatched`; None for a function of the notebook
    or of a library."""
    namespace = getattr(fn, "__globals__", None)
    module = sys.modules.get(namespace.get("__name__")) if isinstance(namespace, dict) else None
    if not isinstance(module, types.ModuleType) or vars(module) is not namespace:
        return None
    try:
        local = is_local_module(module)
    except (TypeError, AttributeError):
        return None
    return dict(namespace) if local else None


def _rebound_unwatched(call: _Call, before: Mapping[str, Any] | None) -> bool:
    """Whether the call rebound a global of its module that the entry does
    not capture (``globals()[name] = v``, ``setattr(sys.modules[__name__],
    ...)``): a hit would return the result and leave the module as it was,
    so the site is refused, as for a draw or a changed argument. The
    globals its code says it writes (``mutated_globals``) are captured and
    put back on a hit."""
    if before is None:
        return False
    now = call.fn.__globals__
    watched = set(call.mutated_globals) | INTERPRETER_MANAGED_GLOBALS
    for name in now.keys() | before.keys():
        if name not in watched and now.get(name, _ABSENT) is not before.get(name, _ABSENT):
            return True
    return False


_ABSENT = object()
