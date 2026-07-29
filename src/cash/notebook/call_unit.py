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

import hashlib

from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.call_interception import CallSite
from cash.notebook.object_hashing import compute_hash

__all__ = ["call_cache_key"]


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
