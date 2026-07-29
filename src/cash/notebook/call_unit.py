from __future__ import annotations

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

import ast
import functools
import hashlib
import logging
import sys
import time as _time
from collections.abc import Callable, Mapping
from typing import Any

from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cacheability import analyze_statement
from cash.notebook.cacheability_decision import decide_cacheability
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.call_interception import CallSite, _names_read
from cash.notebook.file_tracker import FileAccessTracker
from cash.notebook.object_hashing import compute_hash, is_identity_fallback_hash
from cash.notebook.randomness import capture_rng_state, rng_modules_changed

logger = logging.getLogger(__name__)

__all__ = ["call_cache_key", "call_site_is_cacheable", "CallUnit"]


def call_site_is_cacheable(
    call_node: ast.Call,
    *,
    user_ns: Mapping[str, Any],
    annotation: CacheAnnotation | None,
    is_stateful_call: Callable[[str], bool],
    scan_forbidden: Callable[[str, Mapping[str, Any], ast.Module | None], list[str]],
    variable_lineage: Mapping[str, str] | None = None,
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
    """
    tree = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
    code = ast.unparse(call_node)
    inputs = _names_read(call_node) if variable_lineage is not None else set()
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


def call_cache_key(
    site: CallSite,
    *,
    ctx: CacheKeyContext,
    arg_digests: list[str],
    loop_vars: dict[str, object],
) -> str | None:
    """The cache key for one intercepted call, or ``None`` to refuse caching it.

    Delegates to the canonical builder (:func:`compute_cache_key`, ADR-007's
    only key assembler) with the *call's* source and free names in place of
    the statement's, under the ``"call"`` namespace. Three things fall out of
    that for free, because the callee name is itself one of the free names:

    * editing the callee re-keys the call (``func_source_hashes``),
    * globals the callee reaches at call time are folded in
      (``called_function_dependencies``), including ones bound below,
    * bare-``ast.Name`` arguments resolve through the lineage ladder — a dict
      lookup, so a loop-invariant ``big_df`` costs nothing per iteration,
      which is the whole reason this is not the decorator.

    **Hybrid keying.** That lineage resolution is not enough on its own: every
    argument expression that is NOT a bare ``ast.Name`` contributes a content
    hash of its *evaluated value*, supplied by the caller as *arg_digests*
    (the thunk holds the live arguments). That second half is not an
    optimisation, it is a correctness requirement. ``compute(next(it))`` reads
    ``it``, whose lineage is an id-based hash that never moves as the
    iterator is consumed (``it`` fails ``pickle.dumps``, so ``compute_hash``
    falls back to ``sha256(str(id(obj)))``) — so a pure-lineage key would
    collapse every iteration onto one entry and serve iteration 1's value for
    all of them, wrong on the first run with no cache pre-existing.

    Both *arg_digests* and *loop_vars* are REQUIRED, with no default. A
    default would let a caller silently omit the only discriminator a call
    has, producing a collapsed key with no error and no failing test — exactly
    the bug this function exists to prevent.

    ``computed_arg_positions`` records exactly which argument positions are
    NOT bare Names, so ``len(arg_digests)`` must equal
    ``len(site.computed_arg_positions)``. A mismatch means the caller has lost
    the only discriminator those arguments have, and minting a key anyway
    would risk a collapsed, wrong one — so this refuses (returns ``None``)
    rather than guess. A caching optimisation must never be why user code
    fails: an uncached call is merely slow, a wrong cached value is silently
    incorrect. Task 5's thunk must treat ``None`` as "run uncached".

    **loop_vars** are the non-dunder entries of the enclosing iteration
    context (``for_handler.py:278``) — the loop variable's *value*, empty
    outside a loop. They close the remaining channel: hidden state behind a
    bare Name (``fetch_next(conn)``), where no argument expression exists to
    hash and the lineage never moves. That "non-dunder" restriction is
    enforced HERE, not merely documented and trusted to the caller: any
    dunder-prefixed entry in *loop_vars* is filtered out before hashing.
    ``for_handler``'s iteration context carries a dunder half too
    (``__iterable_lineage__``, the whole iterable's lineage — the CAS-242
    culprit) alongside the loop variable's value, and if a caller ever passed
    that whole context through unfiltered it would be hashed in like any
    other entry and CAS-242 would be back. Three properties, and all three
    are needed:

    * they discriminate iterations, which is what the omitted iteration
      context used to do;
    * they are order-independent — item ``5``'s value is ``5`` whatever
      position it occupies — unlike ``__iterable_lineage__``, which changes
      for every iteration on a reorder and is the whole of CAS-242;
    * they are stable across runs, so restoring the earlier iterations and
      re-running only the tail keys correctly. A per-run execution counter
      was specified in an earlier draft (``repeat_index``) and fails exactly
      here: it restarts at 0 each run, so under partial tail re-execution the
      new iteration keys ``rpt0`` and is served run 1's *first* value — wrong,
      in the workflow this feature exists for. It has been removed.

    Note that duplicate loop items collapse to one key, and that is CORRECT —
    same callee, same inputs, no dependency cash cannot see. The statement
    path already collapses them: ``compute_context_hash`` yields one hash for
    three identical iteration contexts, so this matches shipped behaviour.

    **What is deliberately NOT here: the iteration context.** ``for_handler``
    prepends ``# __iteration_context__: <hash>`` to each body statement, and
    that context carries ``__iterable_lineage__`` — so reordering a loop's
    iterable changes the source hash of *every* iteration and re-runs the
    whole tail. That is CAS-242. A call keyed on its own source, its own free
    variables, and the loop variable's *value* (not the iterable's lineage)
    has no such comment to inherit, which is precisely why this fixes it.
    **Do not "fix" a cache miss by adding the iteration context here.**

    Note: ``rng_fingerprint`` is deliberately not threaded through — it is
    dead plumbing with no producer anywhere in the codebase.
    """
    base = compute_cache_key(
        site.source,
        set(site.free_names),
        ctx=ctx,
        occurrence_index=site.occurrence_index,
        namespace="call",
    ).cache_key
    # Refuse rather than mint a key we cannot justify. `computed_arg_positions`
    # records exactly which arguments are NOT bare Names, so a caller that
    # supplies the wrong number of digests has lost the only discriminator
    # those arguments have -- and a collapsed key is a first-run wrong answer,
    # while an uncached call is merely slow.
    if len(arg_digests) != len(site.computed_arg_positions):
        return None

    # Dunder keys are filtered HERE, not trusted to the caller. `loop_vars` is
    # documented as the non-dunder entries of the iteration context, and the
    # dunder half is `__iterable_lineage__` -- the whole iterable's lineage,
    # which changes for every iteration on a reorder. If it ever reached this
    # function unfiltered it would be hashed in like any other entry and
    # CAS-242 would be back. Enforce the contract rather than documenting it.
    filtered_loop_vars = {
        name: value for name, value in loop_vars.items() if not name.startswith("__")
    }

    if not arg_digests and not filtered_loop_vars:
        return base
    # Length-prefixed and `|`-delimited deliberately: `":".join(["a", "b"])`
    # and `":".join(["a:b"])` are the same string, so an undelimited join
    # would let two different calls collide on one key.
    parts = [f"{len(arg_digests)}"]
    parts.extend(arg_digests)
    parts.extend(
        f"{name}={compute_hash(value)}"
        for name, value in sorted(filtered_loop_vars.items())
    )
    return "call:" + hashlib.sha256(
        (base + "|" + "|".join(parts)).encode("utf-8")
    ).hexdigest()


#: Below this, a call is not worth a key, a store, or a timer -- mirrors the
#: statement path's ``min_execution_time_to_cache_seconds`` floor
#: (``statement/processor.py:_store_in_cache``, default 0.01s). This is what
#: makes an allow-list of "trivial builtins" unnecessary: ``len()`` can never
#: clear it.
_COST_FLOOR_S = 0.010


class _ForwardingTee:
    """Records everything written while forwarding untouched to the real stream.

    Used by :meth:`CallUnit._call_capturing_output` to capture a callee's own
    stdout/stderr for later replay on a cache hit, without disturbing the
    statement's own ambient capture: *real_stream* IS that ambient capture's
    current stdout/stderr object during a miss, so every write still reaches
    it exactly as before this class existed. Deliberately not the processor's
    own ``_TeeWriter`` (``statement/processor.py``) -- this module sits
    beneath the processor in the import graph and must not depend on it.

    **Known gap, not fixed here**: ``sys.stdout.buffer`` (the underlying
    binary stream some libraries write raw bytes to directly, bypassing the
    text layer) is forwarded untouched by ``__getattr__`` and is NOT
    recorded -- a callee writing through it produces no replay text on a
    later hit. Recording binary writes would need a second, byte-oriented
    tee wired through ``.buffer`` specifically; text ``write``/``writelines``
    (what ``print`` and the overwhelming majority of callees use) are the
    channels this class covers.
    """
    __slots__ = ("_real", "_chunks")

    def __init__(self, real_stream: Any) -> None:
        self._real = real_stream
        self._chunks: list[str] = []

    def write(self, s: str) -> int:
        self._real.write(s)
        self._chunks.append(s)
        return len(s)

    def writelines(self, lines) -> None:
        # Not delegated to ``__getattr__``: routing straight to
        # ``self._real.writelines`` would bypass ``_chunks`` entirely, so a
        # callee using ``sys.stdout.writelines([...])`` (or a library that
        # does under the hood) produced empty replay text on a later hit.
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._real.flush()

    def getvalue(self) -> str:
        return "".join(self._chunks)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


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

    def __init__(self, cash_instance, ctx_provider: Callable[[], CacheKeyContext]):
        self._cash = cash_instance
        self._ctx_provider = ctx_provider
        self.call_log: list[dict] = []
        #: Cache keys of sites known to mutate an argument or consume RNG,
        #: discovered by observing a MISS (see `wrap`). Permanent for the life
        #: of this `CallUnit` (one notebook session): once a site is known to
        #: have an effect this feature cannot replay, it must never be served
        #: or written again, on any later call -- including one wrapped by a
        #: fresh `wrap()` call for the same site (`CallCache.resolve` re-wraps
        #: per statement execution), which is why this lives on `self` and not
        #: on `_invoke`'s closure.
        self._refused: set[str] = set()

    def wrap(self, fn, site: CallSite):
        func_name = self._func_name(fn)

        @functools.wraps(fn)
        def _invoke(*args, **kwargs):
            key = self._build_key(site, args, kwargs)
            if key is None:
                # Either the key build raised, or `call_cache_key` itself
                # refused (arg-digest count mismatch, by design). Either way
                # this call is not safely keyable right now -- run it
                # uncached rather than risk a collapsed, wrong key.
                return fn(*args, **kwargs)

            if key in self._refused:
                # A previous miss on this exact site proved its effects
                # cannot be replayed (argument mutation or an RNG draw). Run
                # it plain -- never look it up, never store over it.
                return fn(*args, **kwargs)

            hit, value, recorded_cost, metadata = self._lookup(key)
            if hit:
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
                self._record(func_name, site, key, cache_hit=True, elapsed=0.0, time_saved=recorded_cost)
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
            # recorded here -- Task 6b already refuses any call that consumed
            # it (CAS-254 tracks a proper fix).
            #
            # A NESTED tracker, not a before/after diff against the ambient
            # (statement-wide) one -- ``core.py:1870`` is what this task was
            # told to copy, and it wraps the DECORATED CALL in its own fresh
            # ``FileAccessTracker(propagate_to_parent=True)``, not a diff
            # against the caller's tracker. That distinction is load-bearing:
            # ``_active_tracker.get()`` is shared for the whole statement (or,
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
            started = _time.perf_counter()
            call_tracker = FileAccessTracker(
                getattr(fn, '__globals__', None), propagate_to_parent=True,
            )
            with call_tracker:
                result, stdout_text, stderr_text = self._call_capturing_output(fn, args, kwargs)
            elapsed = _time.perf_counter() - started

            if rng_modules_changed(rng_before, capture_rng_state()):
                # RNG is a consumed linear resource -- what matters is stream
                # POSITION, not membership. A hit leaves the global stream
                # where it was, so every downstream draw would diverge from
                # the uncached oracle. Replaying it properly needs a
                # sub-statement position anchor that does not exist yet
                # (CAS-254); v1 refuses instead of guessing.
                self._refused.add(key)
            elif self._hash_args(args, kwargs) != arg_hashes_before:
                # The callee mutated a live argument in place and returned
                # something else (`df.dropna(inplace=True); return len(df)`).
                # Task 6's identity check only catches `return arg` -- this
                # catches "mutated but returned a *different* object", which
                # a hit would silently skip.
                self._refused.add(key)
            elif elapsed >= _COST_FLOOR_S and self._storable(result, args, kwargs):
                self._store(
                    key, result, elapsed,
                    file_deps=frozenset(call_tracker.get_accessed_files()),
                    remote_deps=frozenset(call_tracker.get_accessed_remote_urls()),
                    stdout=stdout_text,
                    stderr=stderr_text,
                )
            self._record(func_name, site, key, cache_hit=False, elapsed=elapsed)
            return result

        return _invoke

    def _call_capturing_output(self, fn, args: tuple, kwargs: dict) -> tuple[Any, str, str]:
        """Run *fn*, returning ``(result, stdout_text, stderr_text)``.

        Tees ``sys.stdout``/``sys.stderr`` through a recorder that still
        forwards every byte to the stream that was live going in -- which,
        during a real statement execution, IS the statement's own ambient
        capture (a ``StringIO``, a ``_TeeWriter``, or the real terminal
        outside any capture). So a genuine miss looks exactly as it did
        before this method existed: the callee's output reaches the
        statement's capture "for free", untouched.
        The recording exists only so a LATER hit (:meth:`_replay_output`) has
        something to write back onto the live stream -- otherwise that
        output is simply gone, since the callee does not run at all on a hit.
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee_out = _ForwardingTee(old_stdout)
        tee_err = _ForwardingTee(old_stderr)
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
        the sub-call started hitting -- exactly the CAS-243 regression this
        task exists to close.
        """
        snap = metadata.get("auto_file_deps")
        if not snap:
            return
        try:
            from cash.notebook.file_tracker import _active_tracker
            tracker = _active_tracker.get()
        except Exception:  # noqa: BLE001 - tracking is best-effort
            return
        if tracker is None:
            return
        for path, recorded in snap.items():
            try:
                # A remote entry must go back onto the remote channel --
                # routed to ``_add_tracked`` it would enter the file set, be
                # stat'ed, and be dropped, same reasoning as
                # ``core.py``'s ``_propagate_file_deps_to_active_tracker``.
                if isinstance(recorded, dict) and recorded.get("remote"):
                    tracker._add_tracked_remote(path)
                else:
                    tracker._add_tracked(path)
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
            except Exception:  # noqa: BLE001
                logger.debug("call unit: could not replay stdout")
        if stderr_text:
            try:
                sys.stderr.write(stderr_text)
            except Exception:  # noqa: BLE001
                logger.debug("call unit: could not replay stderr")

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
            except Exception:  # noqa: BLE001 - belt: compute_hash's own tier-3
                # fallback (`identity_hash`) hashes `id(obj)`, which cannot
                # itself raise, so this except is not expected to be live in
                # practice. Kept because "cannot prove unmutated" is exactly
                # what a `None` here already means to the caller, and it costs
                # nothing to keep the same fail-closed answer if that ever
                # stops being true.
                out.append(None)
                continue
            if is_identity_fallback_hash(value, h):
                out.append(object())
            else:
                out.append(h)
        return tuple(out)

    def _build_key(self, site: CallSite, args: tuple, kwargs: dict) -> str | None:
        if site.has_unpacking:
            # `*args`/`**kwargs` unpacking means the call's live arity is not
            # statically known. `site.computed_arg_positions` is a STATIC
            # count (every position, fail-closed -- see
            # `_computed_arg_positions`), which need not match the RUNTIME
            # flattened `(*args, *kwargs.values())` length: `compute(*pair())`
            # has one static position but the pair unpacks to two live
            # arguments, and indexing only position 0 would hash the first
            # element and silently ignore the rest (CAS-243 review C2 --
            # reproduced as a second, DIFFERENT pair() result being served the
            # first call's cached value). Refuse the whole site rather than
            # mint a key that looks discriminated but isn't; an uncached call
            # is merely slow.
            return None
        try:
            arg_digests = self._arg_digests(site, args, kwargs)
            return call_cache_key(
                site,
                ctx=self._ctx_provider(),
                arg_digests=arg_digests,
                # TODO(CAS-243): the enclosing iteration context's non-dunder
                # entries (`for_handler.py:278`'s `loop_vars`) are not reachable
                # from here. `_process_one_iteration` computes them and threads
                # them down to the statement body via source-text markers and
                # `self.shell.user_ns`, neither of which carries live loop-var
                # VALUES into an exec'd call's closure -- there is no
                # user_ns/thread-local/module-global carrying "the current
                # iteration's loop variable" today. Passing `{}` unconditionally
                # is therefore correct-but-degraded for `fetch_next(conn)`-shaped
                # hidden-state calls inside a loop (no argument expression exists
                # to hash and the lineage never moves) -- see the "loop_vars"
                # section of `call_cache_key`'s docstring. A real fix needs a
                # stack-shaped attribute on the statement processor, set/reset by
                # `for_handler.py` around each body statement's execution, and
                # read here through `ctx_provider`'s owner. Flagged, not faked.
                loop_vars={},
            )
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _arg_digests(self, site: CallSite, args: tuple, kwargs: dict) -> list[str]:
        """Content hashes of the live arguments at ``site.computed_arg_positions``.

        Positions are in ``(*args, *kwargs.values())`` order, matching how
        :func:`_computed_arg_positions` numbered them at rewrite time. A
        position beyond the live call's arity (the wrapped function called with
        a different shape than the site predicted) is simply not appended --
        the resulting length mismatch is caught by ``call_cache_key`` itself,
        which refuses rather than mint a key with a discriminator missing.
        """
        combined = (*args, *kwargs.values())
        digests = []
        for pos in site.computed_arg_positions:
            if pos >= len(combined):
                continue
            digests.append(compute_hash(combined[pos]))
        return digests

    def _storable(self, result, args, kwargs) -> bool:
        """Refuse values whose *identity* is load-bearing.

        Two families, both of which the statement path already refuses in its
        own vocabulary:

        1. **The result IS one of the arguments.** ``def f(d): d['k']=1;
           return d`` -- a hit would hand back a deserialised copy, so
           ``a = f(d)`` gives ``a is not d`` where Python guarantees identity.
           The statement path's alias rule only reaches a bare bind
           (``b = a``); CAS-170 records the computed-RHS version as
           structurally unfixable per-statement. At the call node the live
           arguments are in hand, so it is one ``is`` check.

        2. **Identity-coupled library objects** -- a matplotlib Figure/Axes is
           only correct while it IS the object pyplot's registry points at.
           The RAM tier deep-copies on store and ``Figure.__setstate__``
           re-registers the COPY as the current figure, so a later bare
           ``plt.savefig()`` writes the cache's snapshot. Refusing here lands
           BEFORE the write, which is what stops the copy being made at all.

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
        for arg in args:
            if result is arg:
                return False
        for arg in kwargs.values():
            if result is arg:
                return False
        try:
            from .cacheability_decision import identity_coupled_reason
            return identity_coupled_reason("<intercepted call>", result) is None
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
        re-validates it, exactly mirroring ``Cash._auto_file_deps_fresh``
        (``core.py``) -- a stale entry is treated as a miss like any other,
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
        try:
            cost = float(metadata.get("execution_time", 0.0))
        except (TypeError, ValueError, AttributeError):
            cost = 0.0
        return True, value, cost, metadata

    @staticmethod
    def _auto_file_deps_fresh(metadata: Mapping[str, Any]) -> bool:
        """Mirrors ``Cash._auto_file_deps_fresh`` (``core.py``) for a call
        entry's own recorded dependencies.

        Same snapshot shape (``{path: {'mtime', 'size'[, 'hash']}}`` for a
        local file, ``{'remote': True, ...}`` for a remote read -- see
        :mod:`cash.notebook.file_dep_snapshot`) and the same freshness
        helper, so the two subsystems cannot drift on what "fresh" means.
        Absent/empty ``auto_file_deps`` (a call that read no files) is
        vacuously fresh, same as the decorator's version.
        """
        snap = metadata.get("auto_file_deps") or {}
        if not snap:
            return True
        try:
            from cash.notebook.file_dep_snapshot import file_dep_is_fresh
        except Exception:  # noqa: BLE001 - never let a broken import fail-open a hit
            return False
        for path, recorded in snap.items():
            try:
                is_fresh, _reason = file_dep_is_fresh(path, recorded)
            except Exception:  # noqa: BLE001 - fail closed: cannot prove fresh
                return False
            if not is_fresh:
                return False
        return True

    def _store(
        self, key: str, value, elapsed: float, *,
        file_deps: frozenset[str] = frozenset(),
        remote_deps: frozenset[str] = frozenset(),
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Write through ``backend.set(key, value, metadata)`` -- the same
        two-positional-argument shape the statement path uses
        (``_store_in_cache``), not a merged single-dict entry.

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
        metadata: dict[str, Any] = {"execution_time": elapsed, "timestamp": _time.time()}
        if file_deps or remote_deps:
            try:
                from cash.notebook.file_dep_snapshot import snapshot_dependencies
                snap = snapshot_dependencies(file_deps, remote_deps)
            except Exception:  # noqa: BLE001 - never let dep snapshotting break the store
                snap = None
            if snap:
                metadata["auto_file_deps"] = snap
        if stdout:
            metadata["stdout"] = stdout
        if stderr:
            metadata["stderr"] = stderr
        try:
            self._cash.backend.set(key, value, metadata)
        except Exception:  # noqa: BLE001
            logger.debug("call unit: store failed for %s", key)

    def _func_name(self, fn) -> str:
        """Mirrors ``CallCache._log_key`` so ``_mark_intercepted_calls`` (which
        matches drained ``decorator_calls`` entries against
        ``CallCache.wrapped_names`` by this same string) keeps working.
        """
        try:
            return self._cash._get_func_key(fn)
        except Exception:  # noqa: BLE001
            return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"

    def _record(self, func_name, site: CallSite, key, *, cache_hit, elapsed, time_saved=0.0) -> None:
        """Emit the SAME event shape ``drain_decorator_calls`` returns.

        Keeping the contract identical is what lets the badge, the ``@cache``
        row and ``%cash_stats`` keep working on this log unmodified.
        """
        self.call_log.append({
            "func_name": func_name,
            "cache_hit": cache_hit,
            "execution_time": elapsed,
            "time_saved": time_saved,
            "args_hash": "",
            "cache_key": key,
            "timestamp": _time.time(),
            "call_source": site.source,
            "occurrence_index": site.occurrence_index,
        })

    def drain(self) -> list[dict]:
        events, self.call_log = self.call_log, []
        return events
