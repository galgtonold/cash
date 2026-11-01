"""Which input's change made a statement recompute, for the badge."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from cash._user_attrs import attribute_or

from ..lineage_formula import resolve_lineage_alias

if TYPE_CHECKING:
    from ..tracking_state import TrackingState

__all__ = ["input_change_reason", "stored_input_change_reason", "version_slot"]

logger = logging.getLogger(__name__)


def version_slot(source_hash: str, outputs: Iterable[str]) -> str:
    """What makes two stored results versions of one statement: its source and
    the names it binds (``version_slot`` in its metadata). Two cells with the
    same statement share a slot; a version one of them reads in this process
    is never pruned for the other."""
    return hashlib.sha256(f"{source_hash}|{','.join(sorted(outputs))}".encode()).hexdigest()[:32]


def input_change_reason(tracking_state: TrackingState, inputs, outputs) -> str | None:
    """The miss reason naming the input whose change forced a statement to
    recompute, or None.

    An upstream input changing is the most common reason a notebook
    statement re-runs, and it was the one reason the badge could not name:
    the row rendered EXECUTED with no attribution. A user with a
    reproducible slow re-run had nowhere to look but cash's source.

    Cheap by construction, and it must stay that way. ``TrackingState``
    already records, per output variable, the input lineages the statement
    last RAN with -- so the comparison is that record against the current
    lineages, an O(inputs) dict walk. It never touches the backend:
    answering the same question by scanning the cache is O(N^2) in cache
    size over a run and dominates cold-run wall time.

    Ordering is load-bearing: ``executed_input_lineages`` is rewritten by
    the processor's ``_post_execute``, which runs AFTER this. Reading it
    here therefore sees the previous run's inputs, which is the whole point -- once the
    statement has run, its inputs agree again and the reason is gone.

    Silent when it has nothing to say: a first run has no prior record, a
    statement whose inputs all match did not re-run because of them, and a
    name the statement also WRITES is excluded outright -- see the comment
    on ``wanted`` below for why that comparison cannot be trusted.

    A wrong reason is worse than no reason here. The row rendered EXECUTED
    with no attribution before this function existed, so failing closed to
    silence costs a diagnostic; failing open sends the user to inspect a
    variable that is not the problem.
    """
    try:
        state = tracking_state
        current = state.variable_lineage
        # A name this statement WRITES is not evidence about what it read.
        # ``executed_input_lineages`` is keyed by output variable name
        # alone, so every statement writing the same variable shares one
        # slot and reads back whichever of them ran last. For a chain of
        # ``df = df[...]`` filters that is always a different statement,
        # and the mismatch is guaranteed -- 17 of 21 attributions on
        # 01_nyc_taxi_analysis named an input that was also the
        # statement's own output, on a run whose cache keys and lineage
        # sequence were byte-identical to the previous one.
        #
        # Dropping only the self-referential names keeps the reason for
        # the half of the statement that is still sound:
        # ``df = df.join(other)`` may honestly blame ``other``.
        wanted = {v for v in (inputs or []) if isinstance(v, str) and v not in set(outputs or [])}
        for out in outputs or []:
            previous = state.executed_input_lineages.get(out)
            if not previous:
                continue
            stale = sorted(
                name for name, was in previous.items() if name in wanted and name in current and current[name] != was
            )
            if stale:
                names = ", ".join(stale[:3])
                more = f" +{len(stale) - 3} more" if len(stale) > 3 else ""
                return f"input changed: {names}{more}{_budget_cut(state, stale, current)}"
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        return None
    return None


def _budget_cut(state: TrackingState, stale: list[str], current) -> str:
    """Why the first of *stale* that counts as changed because its change
    check ran out of its budget (`value_hash.budgeted_hash`) changed, or
    "": not read through, it may well be what it was."""
    cuts = attribute_or(state, "budget_cut_lineages", None) or {}
    for name in stale:
        found = cuts.get(name)
        if found is not None and found[0] == current.get(name):
            return (
                f" ({name}'s check for a change in place gave up after {found[1]:.1f}s, "
                "too costly to read, so it counts as changed)"
            )
    return ""


def stored_input_change_reason(
    backend: Any,
    cache_key: str,
    source_hash: str,
    inputs: Iterable[str],
    outputs: Iterable[str],
    current: Mapping[str, str],
) -> str | None:
    """The miss reason naming the inputs whose lineage differs from the one
    the statement's last stored version was built from, or None.

    `input_change_reason` reads what this session ran; after a restart it
    has nothing, and a slow statement whose key changed because an input's
    lineage did ran again with no reason given. The version stored last
    (`version_slot`, ``CacheBackend.previous_version``) keeps the lineages
    of its inputs (``input_lineages``): one metadata read on a miss that
    runs the statement anyway. Silent when no version is kept, or when no
    input differs (the key changed for another reason)."""
    outputs = set(outputs)
    wanted = {name for name in inputs if isinstance(name, str) and name not in outputs}
    if not cache_key or not wanted:
        return None
    try:
        previous = backend.previous_version(version_slot(source_hash, outputs), cache_key)
        metadata = backend.peek_metadata(previous) if previous is not None else None
    except Exception:  # a miss reason is a diagnostic
        logger.debug("Could not read the version stored before", exc_info=True)
        return None
    was = (metadata or {}).get("input_lineages")
    if not isinstance(was, Mapping):
        return None
    stale = sorted(
        name
        for name in wanted
        if isinstance(was.get(name), str)
        and name in current
        and resolve_lineage_alias(current[name]) != resolve_lineage_alias(was[name])
    )
    if not stale:
        return None
    names = ", ".join(stale[:3])
    more = f" +{len(stale) - 3} more" if len(stale) > 3 else ""
    return f"input changed since it was last stored: {names}{more}"
