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
import datetime as _dt
import decimal as _decimal
import dis as _dis
import fractions as _fractions
import functools
import hashlib
import inspect as _inspect
import logging
import pathlib as _pathlib
import sys
import time as _time
import types as _types
import uuid
import warnings
from collections.abc import Callable, Mapping
from types import ModuleType as _ModuleType
from typing import Any

from cash._clock import perf_counter as _perf_counter
from cash._sizing import pandas_nbytes, pickled_size_estimate
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.cacheability import analyze_statement, callee_source_global_mutations
from cash.analysis.cacheability_decision import decide_cacheability, identity_coupled_reason
from cash.backends.value_policy import worth_its_bytes
from cash.notebook._trace import trace_event
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.call_interception import CallSite, names_read
from cash.notebook.call_refs import (
    DIGEST_FIELD,
    ESTIMATED_FIELD,
    SIZE_FIELD,
    UNHASHED_PREFIX,
    digest_and_size,
)
from cash.object_hashing import (
    compute_hash,
    compute_hash_full,
    estimate_object_size,
    is_identity_fallback_hash,
)
from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_dependencies
from cash.tracking.file_tracker import FileAccessTracker, active_tracker, is_user_file
from cash.tracking.randomness import capture_rng_state, rng_modules_changed

from ..cost_model import estimated_restore_time

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


def _is_dunder_loop_var(name: str) -> bool:
    """True if *name*'s own bare portion is dunder-prefixed (CAS-257).

    Handles both shapes `loop_vars` can arrive in: a bare name (`"x"` --
    every direct caller/test that predates CAS-257's depth-keying, e.g.
    `test_call_unit_key.py`'s hand-built dicts) and a depth-prefixed one
    (`"0:x"` -- the production path,
    `StatementProcessor.current_loop_vars_for_call_key`). A depth-prefixed
    dunder (`"0:__iterable_lineage__"`) no longer starts with `"__"` itself
    once the prefix is on -- checking the combined string, as this used to,
    would silently stop catching it and turn the enforced CAS-242 guard back
    into a merely documented one. Splitting off the depth at the first colon
    before checking is what keeps the guard live for both shapes.
    """
    _, _, bare = name.partition(":")
    return (bare or name).startswith("__")


def _loop_var_digest(name: str, value: object, loop_var_digests: Mapping[str, str]) -> str:
    """The discriminating hash for one loop-var entry -- full, never sampled,
    and a dict lookup whenever possible instead of a fresh content hash.

    **Why full, never sampled.** `compute_hash` (the sampling hash used
    elsewhere in this module, e.g. `_hash_args`) reduces any list/tuple over
    200 elements to its first 5 + last 5 (`object_hashing._hash_collection`).
    Two long loop items that agree on both ends but differ in the middle hash
    EQUAL under it while being genuinely different values -- so a loop var
    hashed with `compute_hash` can collapse two iterations onto one key, and
    the second is served the first's cached value. First-run wrongness, no
    pre-existing cache required -- the exact failure `loop_vars` exists to
    prevent, just reached through this leg instead of the missing-loop_vars
    one. `for_handler.py` already learned this lesson for the loop
    variable's own lineage (`_process_one_iteration`): "a sampled hash keyed
    two iterations over arrays that agreed in the sample onto ONE entry -
    wrong result on the first run." `_hash_args`'s sampling stays as-is on
    purpose -- it is a per-call mutation smoke test on a possibly-large live
    argument, a documented, coarser trade that is fine to get occasionally
    wrong in the direction of "assume unmutated." A loop var IS the
    per-iteration discriminator; it has no such slack.

    **Why prefer `loop_var_digests`, not just "full hash it here."** A fresh
    `compute_hash_full(value)` call here is correct but was measured too
    expensive for what it buys: a 200k-row DataFrame costs 0.15ms sampled vs
    19ms full; a 5M-float ndarray 0.03ms vs 14.7ms; a 1M-int list 0.01ms vs
    12.4ms. Against `_COST_FLOOR_S` (3ms, the bar a call's own execution
    time must clear to be worth caching at all), the KEY for a large loop var
    can cost more than the call being decided about. It is also PER CALL, not
    per iteration -- the digest is read fresh inside `CallUnit._build_key`,
    which runs once per intercepted call, so an iteration with N cached calls
    against the same large loop var would pay the full hash N times over.

    `for_handler.py` already pays this exact cost, once per iteration, for a
    different reason: `_process_one_iteration` computes `h =
    val._cash_lineage_hash if hasattr(val, '_cash_lineage_hash') else
    compute_hash_full(val)` for every loop-target binding, at the moment it
    binds the value, before any body statement in that iteration runs. That
    is the SAME digest this function would otherwise recompute -- reusing it
    turns the common case back into a dict lookup, restoring the design's
    actual selling point for this leg (see the module docstring's "bare-Name
    arguments resolve through the lineage ladder" point), with IDENTICAL
    discrimination, because it is the same full hash, shared instead of
    repeated.

    **Why `loop_var_digests`, a value pushed through `loop_vars_scope`, and
    NOT `ctx.variable_lineage`.** A first version of this fix (now reverted)
    looked the digest up in `variable_lineage` instead -- cheap the same way,
    but WRONG: `variable_lineage` is a flat dict keyed only by name, written
    by `for_handler.py` once per iteration and never popped. A nested loop
    reusing the outer loop's target name overwrites the entry for the whole
    remainder of the outer iteration, and nothing restores it when the inner
    loop finishes -- `for t in A: for t in B: pass; call(...)` reads the
    INNER loop's last `t` for a call that runs after the inner loop has
    already ended, back in the OUTER iteration. First-run wrongness, found
    live via a real-kernel repro. `loop_var_digests` instead travels through
    `StatementProcessor.loop_vars_scope`'s push/pop stack -- the SAME stack
    `loop_vars` (values) already uses, which correctly nests because it is
    popped when an iteration's body finishes, restoring whatever level was
    beneath it. Sourcing the digest from that stack, rather than from a
    dict with no scope discipline, is what fixes the staleness without
    reintroducing the cost this whole leg exists to avoid.

    **The fallback.** A name absent from `loop_var_digests` (a loop var whose
    binding didn't go through `for_handler.py`'s own per-iteration push, or
    one bound by an ancestor loop several levels up whose own digest wasn't
    carried this far down -- see `StatementProcessor._depth_keyed_loop_scope`)
    computes `compute_hash_full(value)` directly. This MUST stay the full
    hash. Do not "simplify" it to `compute_hash` -- that would silently
    reintroduce the exact sampled-collision bug this function exists to
    prevent, only for callers that happen to miss the fast path, which is a
    worse failure mode than never having the fast path at all (wrong
    occasionally and quietly, instead of slow always).
    """
    digest = loop_var_digests.get(name)
    return digest if digest is not None else compute_hash_full(value)


#: ``fn.__code__ -> names the body mutates``, before the live-namespace filter.
#:
#: Keyed on the CODE OBJECT, not on a hash of the source, and that is a
#: performance decision with a correctness argument attached.
#:
#: * **Performance.** Reaching a source hash means calling
#:   ``inspect.getsource`` first, and that is the whole cost of this analysis:
#:   measured 74us for a pure callee and 84us for a mutating one, per call,
#:   against a 3ms cost floor -- ~2.5% of the bar a call has to clear to be
#:   worth caching at all, paid on every intercepted call including hits, for a
#:   result that is ``()`` for nearly all of them. Keying on the code object
#:   skips ``getsource`` entirely on the second and later calls.
#: * **Correctness.** A code object is strictly finer than its source: one code
#:   object has exactly one body, and redefining a function in a notebook cell
#:   produces a NEW one, so an edit can never be served the old verdict. It also
#:   sidesteps the question a source hash raises -- identical source meaning
#:   different things in different scopes -- which was answerable here (the
#:   verdict is purely syntactic: "is this name a parameter, a plain local, or
#:   free?" is read off the function's own AST with no reference to a namespace)
#:   but is better not relied upon.
#: * **Why a plain dict is safe.** It holds a strong reference to each key, so a
#:   code object in here cannot be collected and have its ``id`` reused by a
#:   different one -- the failure an ``id()``-keyed memo would have. Growth is
#:   bounded by the number of distinct function versions in a session, i.e. by
#:   how often the user edits a cell, and a code object is a few hundred bytes.
_GLOBAL_MUTATION_CACHE: dict[Any, tuple[str, ...]] = {}


def callee_mutated_globals(fn) -> tuple[str, ...]:
    """Names in *fn*'s own globals that calling *fn* mutates in place (CAS-260).

    The call-unit half of the statement path's
    ``StatementProcessor._callee_mutated_globals``, and deliberately the SAME
    underlying analysis (``cacheability._free_vars_mutated_in_function``) so the
    two paths cannot drift on what counts as a callee's write. The difference is
    only where the source comes from: there, a name resolved against the user
    namespace; here, the live function object already in hand.

    Returned sorted, so the key component built from it is order-stable.

    Memoised on the callee's code object -- see ``_GLOBAL_MUTATION_CACHE`` for
    why that key and not a source hash.

    Never raises. A callee whose source cannot be read (a C builtin, an
    ``exec``'d string, a partial) yields ``()``, which means "no globals to
    capture" -- the pre-existing behaviour, i.e. the write is silently skipped
    on a hit exactly as it is today. That is fail-OPEN, and it is the
    deliberate choice: this feature must never be why user code breaks, and a
    callee cash cannot read is not made safer by refusing to cache it.
    """
    globals_dict = getattr(fn, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return ()
    memo_key = getattr(fn, "__code__", None)
    cached = _GLOBAL_MUTATION_CACHE.get(memo_key) if memo_key is not None else None
    if cached is None:
        try:
            cached = tuple(sorted(callee_source_global_mutations(_inspect.getsource(fn))))
        except Exception:  # noqa: BLE001 - analysis must never break a call
            cached = ()
        if memo_key is not None:
            _GLOBAL_MUTATION_CACHE[memo_key] = cached
    # Filtered per call, not memoised: a name is only capturable while it is
    # actually bound and is not a module. `import` order or a `del` can change
    # that between calls, and the memo above is about the SOURCE, not the
    # namespace.
    return tuple(n for n in cached if n in globals_dict and not isinstance(globals_dict[n], _ModuleType))


def call_cache_key(
    site: CallSite,
    *,
    ctx: CacheKeyContext,
    arg_digests: list[str],
    loop_vars: dict[str, object],
    loop_var_digests: Mapping[str, str] | None = None,
    global_digests: Mapping[str, str] | None = None,
    by_content: bool = False,
    name_digests: Mapping[str, str] | None = None,
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

    **Entry names carry an optional depth prefix (CAS-257 defect 1).** In
    production, ``loop_vars``/``loop_var_digests`` arrive from
    ``StatementProcessor.current_loop_vars_for_call_key`` /
    ``current_loop_var_digests_for_call_key``, whose entries are keyed
    ``"{depth}:{name}"`` rather than bare ``name`` — a name reused by a
    nested loop (``for q in A: for q in B: acc.append(pull(handle))``) would
    otherwise occupy one slot for two different loops' values, silently
    losing the outer scope's discrimination for any call sitting *inside*
    the reuse. Sorting, hashing, and the ``arg_digests``/``loop_var_digests``
    lookups below treat this as an opaque string either way — it only has to
    agree between the two dicts, which the shared provider guarantees. The
    ONE place the prefix is not opaque is the dunder filter immediately
    below (``_is_dunder_loop_var``): a depth-prefixed dunder
    (``"0:__iterable_lineage__"``) no longer starts with ``"__"`` itself, so
    that filter has to look past the prefix rather than at the whole string,
    or the CAS-242 guard it enforces would silently stop firing for the
    production shape.

    **What is deliberately NOT here: the iteration context.** ``for_handler``
    prepends ``# __iteration_context__: <hash>`` to each body statement, and
    that context carries ``__iterable_lineage__`` — so reordering a loop's
    iterable changes the source hash of *every* iteration and re-runs the
    whole tail. That is CAS-242. A call keyed on its own source, its own free
    variables, and the loop variable's *value* (not the iterable's lineage)
    has no such comment to inherit, which is precisely why this fixes it.
    **Do not "fix" a cache miss by adding the iteration context here.**

    **stmt_identity (CAS-256).** The base key above is built from the call's
    OWN source and free names alone, which is silent about which *statement*
    the call sits in. Two different statements whose call text and free names
    happen to agree collapse onto the same base key::

        # cell 2
        for step in ['a', 'b', 'c']:
            vals[step] = fetch_next(conn)      # -> [('a',1), ('b',2), ('c',3)]

        # cell 3
        for step in ['a', 'b', 'c']:
            other[step] = fetch_next(conn)     # served cell 2's values -- WRONG

    Both loops call ``fetch_next(conn)`` with the same free names, the same
    occurrence index (0, each statement starts its own count), and the same
    per-iteration ``loop_vars`` (``step`` takes the same three values) -- so
    without a statement-level discriminator the second loop's every iteration
    hits the first's cache entries. First-run wrongness, no pre-existing cache
    required. ``site.stmt_identity`` (``CallSite``'s docstring has the full
    reasoning for why it is ``ast.unparse`` of the enclosing statement, not
    its raw source text) closes this: it is folded in here as one more
    discriminating component, hashed and delimited exactly like the
    ``arg_digests``/``loop_vars`` components below rather than concatenated
    into ``base``'s own source string, so it cannot collide with them under
    string-join ambiguity. Empty (``""``, the default -- a ``CallSite`` built
    before this field existed, or one where ``wrap_eligible_calls`` could not
    unparse the enclosing statement) contributes nothing, which is exactly
    today's pre-existing behaviour: never a NEW failure mode, only a fix that
    can fail to apply.

    **loop_var_digests** (optional) short-circuits the per-loop-var hashing
    :func:`_loop_var_digest` would otherwise do from scratch: a precomputed,
    already-correctly-scoped ``{name: full_hash}`` map, sourced from
    ``StatementProcessor``'s ``loop_vars_scope`` push/pop stack rather than
    the flat, never-popped ``variable_lineage`` dict (see
    :func:`_loop_var_digest`'s docstring for why that distinction is
    load-bearing, not cosmetic). Omitting it (``None``, the default) is
    always CORRECT, only slower — every entry falls through to a fresh
    ``compute_hash_full(value)`` call, same as before this parameter existed.

    **global_digests (CAS-260)** pins the PRE-call state of every global the
    callee writes, ``{name: full_hash}``. Without it the entry's restored
    post-state would be served over a prefix that never produced it: two calls
    that agree on arguments but enter with a different accumulator get the same
    key, and the second is handed the first's absolute end state. That is the
    partial-accumulator hazard ``cacheability.cacheable_accumulator_loop``'s
    requirement (2) refuses outright, and that CAS-261's split tail had to
    satisfy by keying on the accumulator's post-head lineage.

    ``None``/empty is the shape for every call whose callee writes no globals,
    which is nearly all of them, and it contributes nothing — so an entry
    written before this parameter existed keys identically. Unlike
    ``arg_digests`` there is no count to cross-check against the site, because
    the watch list comes from the CALLEE's source rather than the call's: the
    one caller that populates it (:class:`CallUnit`) derives it from the same
    :func:`callee_mutated_globals` result it uses to capture, so the two cannot
    disagree about which names are covered.

    **by_content** keys the call on what it receives rather than on where its
    arguments came from. The names read only inside computed arguments
    (``site.content_names``) leave the base: their contribution is the
    argument's value, already in *arg_digests* -- which the caller must then
    hash in full. And the statement's identity leaves the key: two statements
    making the same call on the same values get the same result.
    ``fit_score(make_features(cleaned[mid], W))`` was keyed on the lineage of
    all of ``cleaned``, so fixing 35 of 200 machines re-fitted every one --
    495 of 600 on identical features (r23s3) -- and renaming the variable the
    sweep assigns to re-fitted all 180.

    Only the caller can say it is safe, and :class:`CallUnit` says so only
    when nothing the call reads can change without its key changing: every
    argument, loop variable and global the callee reaches is plain data or
    code. ``fetch_next(conn)`` reads a connection whose state moves under a
    fixed lineage -- CAS-256's two statements must keep their own entries.

    **name_digests** (content keying only) are full hashes of arguments passed
    as a bare name, ``{name: digest}``: those names leave the base too. A
    setting passed by name (``fit_series(g, PARAMS, cutoff)``) was keyed on
    its lineage, and ``cutoff = work['date'].max() - 28d`` is rebuilt from
    the frame, so fixing one store re-fitted all 360 (r24s5). The caller
    leaves a large value out, keeping its lineage: hashing a big frame on
    every call would cost more than the dict lookup it replaces.
    """
    free_names = set(site.free_names)
    source, occurrence = site.source, site.occurrence_index
    if by_content:
        free_names -= site.content_names
        free_names -= set(name_digests or ())
        # The call's shape, not its spelling: the computed arguments' values
        # are in the key. And no occurrence: the same call on the same values
        # twice in a cell is the same result, which is what content keying
        # already asserts across statements.
        if getattr(site, "content_source", ""):
            source, occurrence = site.content_source, 0
    base = compute_cache_key(
        source,
        free_names,
        ctx=ctx,
        occurrence_index=occurrence,
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
    #
    # `_is_dunder_loop_var`, not a bare `name.startswith("__")` -- a
    # depth-prefixed key (`"0:__iterable_lineage__"`, CAS-257's production
    # shape) no longer starts with `"__"` itself, so a bare check here would
    # silently stop enforcing this exact guard for the only caller that
    # actually reaches it today.
    filtered_loop_vars = {name: value for name, value in loop_vars.items() if not _is_dunder_loop_var(name)}

    if not arg_digests and not filtered_loop_vars and not site.stmt_identity and not global_digests and not by_content:
        return base
    # Length-prefixed and `|`-delimited deliberately: `":".join(["a", "b"])`
    # and `":".join(["a:b"])` are the same string, so an undelimited join
    # would let two different calls collide on one key.
    parts = [f"{len(arg_digests)}"]
    parts.extend(arg_digests)
    # See `_loop_var_digest`'s docstring: full hash (never sampled) for
    # correctness, preferring a precomputed `loop_var_digests` entry (a dict
    # lookup) over a fresh `compute_hash_full` call for cost -- and NOT
    # `ctx.variable_lineage`, whose lack of scope discipline is what caused
    # the nested-loop staleness bug this parameter exists to fix.
    resolved_digests: Mapping[str, str] = loop_var_digests or {}
    parts.extend(
        f"{name}={_loop_var_digest(name, value, resolved_digests)}"
        for name, value in sorted(filtered_loop_vars.items())
    )
    # The enclosing statement's identity (CAS-256, see this function's own
    # docstring section above and `CallSite.stmt_identity`). Hashed rather
    # than appended raw so an unbounded statement source cannot itself defeat
    # the `|`-delimiting the other parts already rely on.
    # The PRE-call state of the globals this callee writes (CAS-260). Prefixed
    # `g:` so a global named like a loop variable cannot occupy the same slot
    # as that loop var's component under the shared `|`-join.
    if global_digests:
        parts.extend(f"g:{name}={digest}" for name, digest in sorted(global_digests.items()))
    if by_content:
        # Prefixed `n:` so a name cannot share a slot with a loop var or a global.
        parts.extend(f"n:{name}={digest}" for name, digest in sorted((name_digests or {}).items()))
        # Marked, so a key without the statement can never equal one with it.
        parts.append("by=content")
    elif site.stmt_identity:
        parts.append("stmt=" + hashlib.sha256(site.stmt_identity.encode("utf-8")).hexdigest())
    return "call:" + hashlib.sha256((base + "|" + "|".join(parts)).encode("utf-8")).hexdigest()


#: Below this, a call is not worth a key, a store, or a timer.
#:
#: This started as a mirror of the statement path's
#: ``min_execution_time_to_cache_seconds`` (``statement/processor.py:
#: _store_in_cache``, default 0.01s) -- inherited, not measured. The two paths
#: do not have the same overhead, and mirroring made this one over-conservative
#: by ~3x: a whole band of loops cleared neither this floor nor the single-unit
#: threshold and so cached nothing at all.
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


class _ForwardingTee:
    """Records everything written while forwarding untouched to the real stream.

    Used by :meth:`CallUnit._call_capturing_output` to capture a callee's own
    stdout/stderr for later replay on a cache hit, without disturbing the
    statement's own ambient capture: *real_stream* IS that ambient capture's
    current stdout/stderr object during a miss, so every write still reaches
    it exactly as before this class existed. Deliberately not the processor's
    own ``TeeWriter`` (``statement/processor.py``) -- this module sits
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

#: Values whose content is all there is to them: nothing about them can change
#: while their hash stays put. The test :func:`_keys_by_content` applies to
#: everything a call reads before keying it without its statement.
_PLAIN_ATOMS = (
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
    _decimal.Decimal,
    _fractions.Fraction,
    _dt.date,
    _dt.time,
    _dt.timedelta,
    _pathlib.PurePath,
    range,
)
#: How many values one call may have looked at before the answer is "not
#: plain" -- a long list is keyed the old way rather than walked.
_PLAIN_BUDGET = 10_000
#: Computed arguments are hashed in full under content keying; past this many
#: bytes the hash could cost more than the call saves, so the old key stays.
_CONTENT_KEY_MAX_BYTES = 64 * 1024 * 1024
#: An argument passed by name is hashed per call only up to this size; a
#: bigger one keeps its lineage, a dict lookup.
_NAME_CONTENT_MAX_BYTES = 1024 * 1024


def _is_plain_data(value, budget: list[int]) -> bool:
    """True when *value* is data whose content hash covers all of its state.

    Scalars, strings, dates, paths; numpy arrays and scalars without Python
    objects inside; pandas frames, series and indexes (hashed cell by cell);
    builtin containers of those. Anything else -- a connection, an iterator,
    a model, a user class -- may hold state its key cannot see.
    """
    budget[0] -= 1
    if budget[0] < 0:
        return False
    if isinstance(value, _PLAIN_ATOMS):
        return True
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return not value.dtype.hasobject
        if isinstance(value, np.generic):
            return not isinstance(value, np.object_)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(_is_plain_data(item, budget) for item in value)
    if type(value) is dict:
        return all(_is_plain_data(k, budget) and _is_plain_data(v, budget) for k, v in value.items())
    return False


@functools.lru_cache(maxsize=1024)
def _code_names(code: _types.CodeType) -> frozenset[str]:
    """Global names *code* and the functions nested in it load.

    From the bytecode, not ``co_names``, which holds attribute names too:
    ``os.open`` would read as the global ``open`` -- cash's file-tracking
    wrapper in a notebook.
    """
    names = {ins.argval for ins in _dis.get_instructions(code) if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
    for const in code.co_consts:
        if isinstance(const, _types.CodeType):
            names |= _code_names(const)
    return frozenset(names)


def _plain_or_code(value, seen: set[int], budget: list[int]) -> bool:
    if isinstance(value, _types.FunctionType):
        if getattr(value, "_is_file_tracker_patch", False) or getattr(value, "_cash_cached", False):
            return True  # cash's own: keyed or tracked by cash itself

        filename = getattr(value.__code__, "co_filename", "") or ""
        # A cell's code has a `<cash-...>` / `<ipython-...>` name: the user's.
        if filename and not filename.startswith("<") and not is_user_file(filename):
            # A library's function is code, as its classes are. Its module
            # state is no more in a lineage key than in a content key, and
            # walking it refused: sklearn's `normalize` is a validating
            # wrapper whose globals hold non-plain state, so `fit_vectors`
            # re-fitted on byte-identical text (round 25, r25s4).
            return True
        return _callee_state_is_plain(value, seen, budget)
    if isinstance(value, (_ModuleType, type, _types.BuiltinFunctionType)):
        return True
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ufunc):
        return True
    return _is_plain_data(value, budget)


def _callee_state_is_plain(fn, seen: set[int], budget: list[int]) -> bool:
    """True when every global, closure cell and default *fn* can reach --
    through the functions it calls too -- is plain data or code."""
    if id(fn) in seen:
        return True
    seen.add(id(fn))
    namespace = getattr(fn, "__globals__", None) or {}
    for name in _code_names(fn.__code__):
        if name in namespace and not _plain_or_code(namespace[name], seen, budget):
            return False
    for cell in fn.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:  # an empty cell: nothing bound yet
            continue
        if not _plain_or_code(contents, seen, budget):
            return False
    defaults = (*(fn.__defaults__ or ()), *(fn.__kwdefaults__ or {}).values())
    return all(_is_plain_data(value, budget) for value in defaults)


def _nbytes(value) -> int:
    """Roughly how many bytes a full content hash of *value* reads."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray):
        return int(value.nbytes)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        sized = pandas_nbytes(value)
        if sized is not None:
            return int(sized)
        usage = value.memory_usage(index=True, deep=False)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    if isinstance(value, (str, bytes)):
        return len(value)
    return 0


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
    user's own line belonged (round 25, r25s5). Recorded, then re-emitted with
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


def global_names_reached(fn, seen: set[int] | None = None, depth: int = 0) -> set[str]:
    """Global names *fn* loads, and those of the functions it reaches, bounded."""
    seen = set() if seen is None else seen
    code = getattr(fn, "__code__", None)
    if code is None or id(fn) in seen or depth > 6:
        return set()
    seen.add(id(fn))
    names = set(_code_names(code))
    namespace = getattr(fn, "__globals__", None) or {}
    for name in list(names):
        value = namespace.get(name)
        if isinstance(value, _types.FunctionType):
            names |= global_names_reached(value, seen, depth + 1)
        elif isinstance(value, type) and id(value) not in seen:
            # A class the callee builds or calls into: its methods read globals too.
            seen.add(id(value))
            for member in vars(value).values():
                member = getattr(member, "__func__", member)
                if isinstance(member, _types.FunctionType):
                    names |= global_names_reached(member, seen, depth + 1)
    return names


def _loop_vars_the_call_can_read(
    fn, site: CallSite, loop_vars: Mapping[str, object], name_digests: Mapping[str, str] | None
) -> dict[str, object]:
    """The enclosing loops' variables a content-keyed call can still read
    without them being in its key.

    A loop variable is in a call's key to close a channel the arguments do not
    cover: hidden state behind a name, or a global the callee reads. Keyed on
    what it receives, a call's arguments are hashed by value, so a loop
    variable passed in (`make_features(cleaned[mid], win)`) is already there,
    and one the call never reads cannot change its result. Kept anyway, it
    made the sweep's keys differ from the same call outside a loop: scoring
    with the chosen window re-fitted all 200 machines the sweep had just fitted
    (round 25, r25s3). Kept: a loop variable the callee, or a function it
    calls, reads as a global; and one named in the call itself but not hashed
    by value (an argument too big to hash per call). Only a variable the
    arguments carry is dropped: one the call does not mention stays, since
    what the callee can reach is followed through functions and classes but
    not every path (a dispatch table, an object's attribute).
    """
    hashed = set(name_digests or ()) | set(site.content_names)
    reached = None
    kept = {}
    for key, value in loop_vars.items():
        bare = key.split(":", 1)[1] if ":" in key else key
        if bare in hashed:
            if reached is None:
                reached = global_names_reached(fn)
            if bare not in reached:
                continue
        kept[key] = value
    return kept


def _keys_by_content(fn, site: CallSite, args: tuple, kwargs: dict, loop_vars: Mapping[str, object]) -> bool:
    """Whether the call may be keyed on what it receives (see
    :func:`call_cache_key`'s *by_content*).

    Only when nothing it reads can change while its key stays put: every
    argument, every loop variable around it, and everything the callee
    reaches is plain data or code, and the arguments to hash in full are
    small enough to be worth it.
    """
    combined = (*args, *kwargs.values())
    budget = [_PLAIN_BUDGET]
    if not all(_is_plain_data(value, budget) for value in combined):
        return False
    if not all(_is_plain_data(value, budget) for name, value in loop_vars.items() if not _is_dunder_loop_var(name)):
        return False
    computed = sum(_nbytes(combined[pos]) for pos in site.computed_arg_positions if pos < len(combined))
    if computed > _CONTENT_KEY_MAX_BYTES:
        return False
    return _callee_state_is_plain(fn, set(), budget)


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
        self._ctx_provider = ctx_provider
        # The TTL in force for the statement this call sits in (CAS-268).
        # Read at INVOKE time, like `loop_vars_provider`, because one
        # `CallCache` serves every statement and each brings its own
        # annotation. `None` -- the default, and what every direct
        # construction predating this parameter gets -- means "no TTL", which
        # is the behaviour this class had when it ignored TTL entirely.
        self._ttl_provider = ttl_provider or (lambda: None)
        # `# @cash:persist` / `%cash_persist` for the statement this call sits
        # in (CAS-269). Read at STORE time for the same reason `ttl` is read at
        # invoke time: one `CallCache` serves every statement. `False` -- the
        # default, and what every construction predating this parameter gets --
        # leaves the decision to the cost model, which is what this class did
        # when it ignored `persist` entirely.
        self._persist_provider = persist_provider or (lambda: False)
        # See `call_cache_key`'s `loop_vars` section. `None` (rather than
        # requiring every caller to pass one) keeps every existing direct
        # construction of `CallUnit` -- tests included -- working exactly as
        # before: "no loop context available" degrading to `{}`, the same
        # answer this class gave when `loop_vars={}` was hardcoded in
        # `_build_key`.
        self._loop_vars_provider = loop_vars_provider or (lambda: {})
        # See `call_cache_key`'s `loop_var_digests` section and
        # `_loop_var_digest`'s docstring. `None` here is always CORRECT
        # (every entry falls through to a fresh `compute_hash_full`), only
        # slower -- so, same as `_loop_vars_provider` above, every existing
        # direct `CallUnit` construction that predates this parameter keeps
        # working unchanged.
        self._loop_var_digests_provider = loop_var_digests_provider or (lambda: {})
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
        #: ``StatementProcessor._execute_and_drain``).
        self.overhead_s = 0.0
        # Per call SITE: what its key was built from last time, and what moved
        # since -- the badge's answer to "why did this re-run?".
        self._site_parts: dict[tuple, tuple[str, dict]] = {}
        self._site_reason: dict[tuple, str | None] = {}
        # Sites already keyed in THIS statement run. A site called once per
        # loop item keys differently per item BY DESIGN -- that is the loop
        # variable doing its job, not something to explain. Only the first
        # call of each run is compared, against the first call of the last.
        self._keyed_this_run: set = set()
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
        """The bar a call's own execution must clear to be stored.

        Read from config on every decision rather than captured, so
        ``cash.configure(call_cost_floor_seconds=...)`` takes effect
        immediately -- the contract ``min_execution_time_to_cache_seconds``
        already has.

        Checked with isinstance rather than ``float()`` in a try/except: a
        MagicMock's ``__float__`` returns 1.0 instead of raising, and
        ``cash_instance`` is a MagicMock throughout the unit suite, so the
        exception form would silently install a 1-SECOND floor there and cache
        nothing. ``_COST_FLOOR_S`` stays the default and the fallback.
        """
        value = getattr(getattr(self._cash, "config", None), "call_cost_floor_seconds", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return _COST_FLOOR_S

    def begin_cell(self) -> None:
        """A new cell: the results held for references are let go."""
        self.held_results.clear()

    def _hold(self, key: str, value: Any, digest: str | None, size: Any) -> None:
        if digest:
            self.held_results[id(value)] = (value, key, digest, size if isinstance(size, int) else 0)

    def begin_statement(self) -> None:
        """A new statement run: every site starts over (see :meth:`_entry_for`)."""
        self._site_runs.clear()
        self._keyed_this_run.clear()
        self.last_returned = None

    def outermost_result(self) -> tuple[str, int, str] | None:
        """``(call key, id(result), call source)`` of the call that returned
        LAST in this statement -- in ``x = f(g(y))``, ``f``'s -- or ``None``
        when it ran without the cache. The statement uses it to trust a
        reference without re-digesting the value (``call_refs.with_call_refs``)."""
        found = getattr(self, "last_returned", None)
        return found if found and found[0] else None

    def _entry_for(self, fn, site: CallSite, invoke):
        """*invoke* behind the many-cheap-calls guard.

        A call in a comprehension is made once per element, and caching one
        costs a key, a lookup, a store and a file tracker of its own: ~14 ms a
        call around a function reading one small file, 5,030 of them in
        r23s4's ``[read_doc(p) for p in paths]`` -- 8.4 s became 71.6 s. A
        ``for`` loop has ``for_handler._should_execute_loop_as_single_unit``
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
            # cache -- 5220/5225 for a folder of 5,225 files (round 25, r25s4).
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
            # between their cell and their function (round 25, r25s3).
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
                    # the call: a hit pays the key and lookup, then the restore
                    # (round 25, r25s5: 4.4 ms calls, ~4 ms to key).
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
            # CAS-260: globals this callee writes. Resolved per call rather
            # than once per `wrap`, because the underlying source analysis is
            # memoised (`callee_mutated_globals`) while the "is it bound, is it
            # a module" filter genuinely depends on the live namespace. Empty
            # for nearly every callee, and every branch below short-circuits on
            # empty, so an ordinary call pays one memo lookup.
            key_started = _perf_counter()
            mutated_globals = callee_mutated_globals(fn)
            key = self._build_key(
                site,
                args,
                kwargs,
                self._global_digests(fn, mutated_globals) if mutated_globals else None,
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
            # recorded here -- Task 6b already refuses any call that consumed
            # it (CAS-254 tracks a proper fix).
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
            elif (
                elapsed >= self._cost_floor_s()
                and self._storable(result, args, kwargs)
                and self._restore_pays(result, elapsed)
            ):
                # CAS-260: the callee's writes to its own globals, captured as
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
                    tracker.add_tracked(path)
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
    def _global_digests(fn, names: tuple[str, ...]) -> dict[str, str]:
        """PRE-call content hashes of the globals *fn* writes, for the key.

        ``compute_hash_full``, never the sampling ``compute_hash``, for the
        same reason :func:`_loop_var_digest` documents at length: this IS the
        discriminator. ``compute_hash`` reduces a collection over 200 elements
        to its first and last five, so two different accumulator states that
        agree at both ends would key IDENTICALLY -- and an accumulator is
        precisely the shape that grows in the middle. That is first-run
        wrongness, not a missed optimisation.

        The cost this admits is real and bounded by how rare the case is: a
        callee that writes a global at all is uncommon, and the hash is over
        the accumulator, not over the arguments. ``_hash_args``' sampling trade
        is fine where it lives (a coarse per-call mutation smoke test on a
        possibly-huge live argument, allowed to be wrong toward "assume
        unmutated"); it is not fine here.

        A name that cannot be hashed at all is omitted, which makes the key
        LESS discriminating -- so :meth:`_capture_globals` independently
        refuses to store any entry whose capture is not sound, and the pair of
        them fails closed.
        """
        globals_dict = getattr(fn, "__globals__", None) or {}
        digests: dict[str, str] = {}
        for name in names:
            try:
                digests[name] = compute_hash_full(globals_dict[name])
            except Exception:  # noqa: BLE001 - a missing digest only widens the key
                logger.debug("call unit: could not digest global %r", name)
        return digests

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
        existed, so it missed forever -- and the two spellings diverged
        (CAS-246's guard caught it). A copy failure is treated like any other
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

    def _build_key(
        self, site: CallSite, args: tuple, kwargs: dict, global_digests: Mapping[str, str] | None = None, fn=None
    ) -> str | None:
        """The call's key. With *fn*, keyed on what it receives when
        :func:`_keys_by_content` allows it."""
        if site.has_unpacking and fn is not None:
            return self._build_unpacked_key(site, args, kwargs, fn)
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
            loop_vars = self._current_loop_vars()
            by_content = fn is not None and _keys_by_content(fn, site, args, kwargs, loop_vars)
            if getattr(site, "in_loop_unit", False) and not by_content:
                return None
            arg_digests = self._arg_digests(site, args, kwargs, full=by_content)
            name_digests = self._name_digests(site, args, kwargs) if by_content else None
            if by_content and loop_vars:
                loop_vars = _loop_vars_the_call_can_read(fn, site, loop_vars, name_digests)
            ctx = self._ctx_provider()
            key = call_cache_key(
                site,
                ctx=ctx,
                arg_digests=arg_digests,
                loop_vars=loop_vars,
                loop_var_digests=self._current_loop_var_digests(),
                global_digests=global_digests,
                by_content=by_content,
                name_digests=name_digests,
            )
            self._note_key_parts(site, key, ctx, arg_digests, global_digests)
            return key
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _build_unpacked_key(self, site: CallSite, args: tuple, kwargs: dict, fn) -> str | None:
        """The key of a call with ``*``/``**`` unpacking, keyed on what it received.

        ``fit_series(g, **TUNED.get(dept, {}))`` in a comprehension ran uncached
        (r25s5): positions written in the source say nothing about the values
        that arrive, so the site was refused (CAS-243: ``compute(*pair())``
        had keyed the first of two values). What did arrive is in hand here:
        every positional value, and every keyword with its name, hashed in
        full. Only when the call may be keyed on content at all
        (:func:`_keys_by_content`); otherwise refused as before.
        """
        try:
            count = len(args) + len(kwargs)
            received = dataclasses.replace(
                site,
                computed_arg_positions=tuple(range(count)),
                local_arg_positions=tuple(range(count)),
                name_arg_positions=(),
                has_unpacking=False,
            )
            loop_vars = self._current_loop_vars()
            if not _keys_by_content(fn, received, args, kwargs, loop_vars):
                return None
            digests = [compute_hash_full(value) for value in args]
            digests.extend(f"{name}:{compute_hash_full(kwargs[name])}" for name in sorted(kwargs))
            if loop_vars:
                loop_vars = _loop_vars_the_call_can_read(fn, received, loop_vars, None)
            return call_cache_key(
                received,
                ctx=self._ctx_provider(),
                arg_digests=digests,
                loop_vars=loop_vars,
                loop_var_digests=self._current_loop_var_digests(),
                by_content=True,
            )
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _current_loop_vars(self) -> dict[str, object]:
        """The live enclosing loop's non-dunder iteration vars, or ``{}``.

        Wired to ``StatementProcessor.current_loop_vars_for_call_key`` (see that class's
        ``_call_unit_loop_vars`` stack, pushed/popped by
        ``ForLoopHandler._process_one_iteration`` around each iteration's body)
        via ``CallCache``'s ``loop_vars_provider``. Guarded independently of
        ``_build_key``'s own try/except: a provider failure should degrade to
        "no loop discriminator" ``{}`` -- same as running outside a loop --
        not to refusing the key (and therefore the call's caching) entirely.
        """
        try:
            loop_vars = self._loop_vars_provider()
        except Exception:  # noqa: BLE001 - degrade, don't refuse the whole key
            logger.debug("call unit: loop_vars_provider failed for this call")
            return {}
        return loop_vars if isinstance(loop_vars, dict) else {}

    def _current_loop_var_digests(self) -> Mapping[str, str]:
        """The live enclosing loop's precomputed loop-var digests, or ``{}``.

        Wired to ``StatementProcessor.current_loop_var_digests_for_call_key`` (see that
        class's ``_call_unit_loop_var_digests`` stack -- pushed/popped in
        lockstep with ``_call_unit_loop_vars``, by the same
        ``loop_vars_scope`` call) via ``CallCache``'s
        ``loop_var_digests_provider``. Guarded independently of
        ``_build_key``'s own try/except, same reasoning as
        ``_current_loop_vars``: a provider failure degrades to "no
        precomputed digest available" ``{}``, which ``_loop_var_digest``
        treats as "fall through to a fresh `compute_hash_full`" -- slower,
        never wrong -- not to refusing the key entirely.
        """
        try:
            digests = self._loop_var_digests_provider()
        except Exception:  # noqa: BLE001 - degrade, don't refuse the whole key
            logger.debug("call unit: loop_var_digests_provider failed for this call")
            return {}
        return digests if isinstance(digests, Mapping) else {}

    @staticmethod
    def _name_digests(site: CallSite, args: tuple, kwargs: dict) -> dict[str, str]:
        """Full hashes of the small arguments passed as a bare name (see
        :func:`call_cache_key`'s *name_digests*). Past
        ``_NAME_CONTENT_MAX_BYTES`` a value keeps its lineage instead."""
        combined = (*args, *kwargs.values())
        digests = {}
        for name, pos in getattr(site, "name_arg_positions", ()):
            if pos < len(combined) and _nbytes(combined[pos]) <= _NAME_CONTENT_MAX_BYTES:
                digests[name] = compute_hash_full(combined[pos])
        return digests

    def _arg_digests(self, site: CallSite, args: tuple, kwargs: dict, full: bool = False) -> list[str]:
        """Content hashes of the live arguments at ``site.computed_arg_positions``.

        Positions are in ``(*args, *kwargs.values())`` order, matching how
        :func:`_computed_arg_positions` numbered them at rewrite time. A
        position beyond the live call's arity (the wrapped function called with
        a different shape than the site predicted) is simply not appended --
        the resulting length mismatch is caught by ``call_cache_key`` itself,
        which refuses rather than mint a key with a discriminator missing.

        *full* hashes every one of them in full: under content keying the
        value is all the key knows of where the argument came from.
        """
        combined = (*args, *kwargs.values())
        local = set(getattr(site, "local_arg_positions", ()))
        digests = []
        for pos in site.computed_arg_positions:
            if pos >= len(combined):
                continue
            # A comprehension's own variable discriminates its elements, as a
            # loop variable does its iterations: full hash, never sampled.
            hash_fn = compute_hash_full if full or pos in local else compute_hash
            digests.append(hash_fn(combined[pos]))
        return digests

    #: A hit is judged a loss only past this, so timer noise on a cheap call
    #: never refuses it.
    _MIN_HIT_LOSS_S = 0.05

    def _restore_pays(self, result, elapsed: float) -> bool:
        """Whether restoring *result* is predicted to beat computing it again.

        The statement path has refused a value whose predicted restore exceeds
        80% of its compute since the cost model was fitted; the call cache,
        which holds the biggest values in a notebook (round 28, r28s5: 406 MiB
        `net_returns` results), never asked. Predicted for disk -- where it
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
        not fitted on, and r28s5's hits were ~10 s against ~4 s of compute,
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

        Two families, both of which the statement path already refuses in its
        own vocabulary:

        1. **The result IS one of the arguments.** ``def f(d): d['k']=1;
           return d`` -- a hit would hand back a deserialised copy, so
           ``a = f(d)`` gives ``a is not d`` where Python guarantees identity.
           The statement path's alias rule only reaches a bare bind
           (``b = a``); CAS-170 records the computed-RHS version as
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
        if not self._ttl_fresh(metadata):
            return False, None, 0.0, {}
        try:
            cost = float(metadata.get("execution_time", 0.0))
        except (TypeError, ValueError, AttributeError):
            cost = 0.0
        return True, value, cost, metadata

    def _ttl_fresh(self, metadata: Mapping[str, Any]) -> bool:
        """Mirrors ``CacheFreshnessChecker._invalidate_if_ttl_expired``
        (``statement/freshness.py``) for a call entry (CAS-268).

        Before this, ``call_unit.py`` contained no reference to ``ttl`` at all,
        so call entries never expired. Once call interception became the
        default (CAS-243) that quietly hollowed out the annotation: the
        STATEMENT would expire and re-execute while the expensive call inside
        it was still served from an entry with no expiry. Measured on
        ``# @cash:ttl=0`` -- the spelling the docs give for data that must
        never be served stale -- the work did not re-run at all until
        ``# @cash:no-cache-calls`` was added as well.

        Two details are copied deliberately rather than re-derived, because
        both are load-bearing and both are easy to get subtly wrong:

        * ``is not None``, not truthiness. ``ttl=0`` is a REQUEST ("expire
          immediately"), not an absent setting -- the falsy-vs-``None`` slip is
          exactly what CAS-221 was at the statement layer.
        * ``ttl <= 0`` short-circuits without consulting the clock. A
          same-tick re-read can measure ``age == 0.0`` on a coarse timer, and
          ``0.0 > 0`` would hand back the very entry ``ttl=0`` exists to
          reject.

        An entry with no recorded ``timestamp`` reads as age-since-epoch, so it
        expires under any TTL rather than being served forever -- the
        fail-safe direction for a value the caller has asked to keep fresh.
        """
        ttl = self._ttl_provider()
        if ttl is None:
            return True
        if ttl <= 0:
            return False
        try:
            timestamp = float(metadata.get("timestamp") or 0)
        except (TypeError, ValueError):
            timestamp = 0.0
        return (_time.time() - timestamp) <= ttl

    @staticmethod
    def _auto_file_deps_fresh(metadata: Mapping[str, Any]) -> bool:
        """Mirrors ``Cash._auto_file_deps_fresh`` (``core.py``) for a call
        entry's own recorded dependencies.

        Same snapshot shape (``{path: {'mtime', 'size'[, 'hash']}}`` for a
        local file, ``{'remote': True, ...}`` for a remote read -- see
        :mod:`cash.tracking.file_dep_snapshot`) and the same freshness
        helper, so the two subsystems cannot drift on what "fresh" means.
        Absent/empty ``auto_file_deps`` (a call that read no files) is
        vacuously fresh, same as the decorator's version.
        """
        snap = metadata.get("auto_file_deps") or {}
        if not snap:
            return True
        for path, recorded in snap.items():
            try:
                is_fresh, _reason = file_dep_is_fresh(path, recorded)
            except Exception:  # noqa: BLE001 - fail closed: cannot prove fresh
                return False
            if not is_fresh:
                return False
        return True

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
        if function:
            # What `cash inspect` names the entry by: a call key is `call:<sha>`,
            # so every intercepted call used to be listed as "call" (r28s1).
            metadata["function"] = function
        # CAS-269. `TieredBackend` reads exactly this key to bypass the ~0.1s
        # persistence floor, so threading the statement's resolved annotation
        # here is the whole fix -- the statement path writes the same field
        # from the same `force_persist` (`processor._save_to_cache`).
        #
        # Written only when True, keeping the sparse-entry shape every other
        # optional channel here follows, and it is a plain `bool`: metadata is
        # eagerly unpickled for EVERY entry at startup, so nothing but builtins
        # belongs in it (see the `callee_globals` note below).
        if self._persist_provider():
            metadata["force_persist"] = True
        if file_deps or remote_deps:
            try:
                snap = snapshot_dependencies(file_deps, remote_deps)
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
        # r28s5's 1.7 GiB result to learn that took 2.7 s after 2.8 s of
        # compute. It gets a one-off token for a digest and its estimated size
        # (`ESTIMATED_FIELD`): judged for disk on its own, and referred to only
        # by the statement it is the plain result of.
        #
        # Nor for the call a statement is nothing but (``a, b = build()``,
        # `plain_value`): that statement's reference is trusted without a
        # digest, so a token serves -- r28s5's 402 MiB result was worth
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
        # CAS-260. Omitted when empty, like every other optional channel here,
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

    def _note_key_parts(self, site: CallSite, key, ctx, arg_digests, global_digests) -> None:
        """Remember what this call site was keyed on, and what moved since the
        last time it was keyed (read back by :meth:`_why_missed`).

        In-memory and per site, like the statement guard's own components. A
        first run in a fresh kernel has nothing to compare against and says
        nothing -- the same deliberate silence a statement keeps, since "first
        time" is self-evident to someone running a cell for the first time.
        """
        if key is None:
            return
        try:
            site_id = self._site_id(site)
            if site_id in self._keyed_this_run:
                return
            self._keyed_this_run.add(site_id)
            lineages = getattr(ctx, "variable_lineage", None) or {}
            parts = {name: lineages.get(name) for name in site.free_names}
            for i, digest in enumerate(arg_digests or ()):
                parts[f"argument {i + 1}"] = digest
            for name, digest in (global_digests or {}).items():
                parts[name] = digest
            seen = self._site_parts.get(site_id)
            self._site_parts[site_id] = (key, parts)
            if not seen or seen[0] == key:
                self._site_reason.pop(site_id, None)
                return
            moved = sorted(name for name in set(parts) | set(seen[1]) if parts.get(name) != seen[1].get(name))
            # The key moved with no named part of it moving: something else did
            # (a file the callee reads, the callee's own source). Naming
            # nothing beats naming the wrong thing.
            named = ", ".join(moved[:3]) + (", ..." if len(moved) > 3 else "")
            self._site_reason[site_id] = f"changed: {named}" if moved else None
        except Exception:  # noqa: BLE001 - attribution never breaks a call
            logger.debug("call unit: could not note key parts for %s", site.source)

    @staticmethod
    def _site_id(site: CallSite) -> tuple[str, int, str]:
        return (site.source, site.occurrence_index, getattr(site, "stmt_identity", ""))

    def _why_missed(self, site: CallSite) -> str | None:
        """Which named part of this call's key moved since it was last keyed.

        Round 30, r30s4: a sweep re-ran and the badge said only "0/6 hit", so
        the tester had to guess why -- and guessed wrong, then reported the
        re-run as a suspected bug.
        """
        try:
            return self._site_reason.get(self._site_id(site))
        except Exception:  # noqa: BLE001
            return None

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
                "miss_reason": None if cache_hit else self._why_missed(site),
            }
        )

    def drain(self) -> list[dict]:
        events, self.call_log = self.call_log, []
        return events
