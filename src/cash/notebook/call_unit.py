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

__all__ = ["call_cache_key"]


def call_cache_key(
    site: CallSite,
    *,
    ctx: CacheKeyContext,
    arg_digests: list[str] | None = None,
    repeat_index: int = 0,
) -> str:
    """The cache key for one intercepted call.

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

    **repeat_index** is a per-(call site, key) execution counter within one
    cell run. It closes the remaining channel: hidden state behind a bare
    Name (``fetch_next(conn)``), where no argument expression exists to hash.
    Reorder reuse is unaffected — iterations discriminated by argument
    lineage produce distinct keys, so each is seen once and each counts 0.

    **What is deliberately NOT here: the iteration context.** ``for_handler``
    prepends ``# __iteration_context__: <hash>`` to each body statement, and
    that context carries ``__iterable_lineage__`` — so reordering a loop's
    iterable changes the source hash of *every* iteration and re-runs the
    whole tail. That is CAS-242. A call keyed on its own source and its own
    free variables has no such comment to inherit, which is precisely why
    this fixes it. **Do not "fix" a cache miss by adding the iteration
    context here.**

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
    if not arg_digests and not repeat_index:
        return base
    extra = ":".join(arg_digests or []) + f":rpt{repeat_index}"
    return "call:" + hashlib.sha256(
        (base + ":" + extra).encode("utf-8")
    ).hexdigest()
