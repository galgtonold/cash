"""Which input's change made a statement recompute, for the badge."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tracking_state import TrackingState

__all__ = ["input_change_reason"]


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
                return f"input changed: {names}{more}"
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        return None
    return None
