"""A draw from a generator held in a variable, as a change to that variable.

``rng = np.random.default_rng(7)`` in one cell, ``d = {k: draw(k, rng) for k
in range(3)}`` in the next: the second statement moves ``rng`` without binding
it, so nothing gave ``rng`` a new lineage. A statement reading ``rng`` after it
then had the key of one reading the fresh generator, and the two could serve
each other's values: a draw computed from a moved generator came back after a
re-seed, and after a restart.

The statement processor notes where each generator among a statement's inputs
stands before it runs (:func:`carrier_positions`) and which of them moved
(:func:`moved_carrier_names`). Each moved one gets the lineage
:func:`advanced_carrier_lineage` derives from the statement's key, which folds
in its lineage before the draw. A cache hit gives the same lineages from the
names its entry recorded, and the upstream simulation reads that record
(``TrackingState.carrier_advances``, or the entry's metadata after a restart)
to give them too.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ...tracking.randomness import advanced_carrier_lineage, carrier_positions, moved_carrier_names, rng_carrier_kind
from ..cache_key import called_function_globals

if TYPE_CHECKING:
    from ..tracking_state import TrackingState

__all__ = [
    "PAYLOAD_FIELD",
    "advance_carriers",
    "carrier_candidates",
    "carriers_an_entry_advanced",
    "reachable_generators",
]

#: The names a stored statement drew from, in its value entry.
PAYLOAD_FIELD = "rng_carriers_moved"


def carrier_candidates(inputs: Iterable[str], user_ns: dict[str, Any]) -> set[str]:
    """The variables a statement can draw from: its inputs, and the globals
    the functions it calls read (``def draw(k): return rng.normal() + k``),
    which its key folds in too (``called_function_dependencies``)."""
    inputs = set(inputs)
    try:
        return inputs | called_function_globals(inputs, user_ns)
    except (TypeError, ValueError, AttributeError, RecursionError):
        return inputs


def reachable_generators(inputs: Iterable[str], user_ns: dict[str, Any]) -> set[str]:
    """The variables among :func:`carrier_candidates` that hold a random
    generator: what a statement reading *inputs* can draw from."""
    return {name for name in carrier_candidates(inputs, user_ns) if rng_carrier_kind(user_ns.get(name)) is not None}


def advance_carriers(
    tracking_state: TrackingState,
    source_hash: str,
    names: Iterable[str] | None,
    statement_key: str,
    code: str,
    user_ns: dict[str, Any],
) -> None:
    """Give each of *names* the lineage of a generator the statement *code*,
    keyed *statement_key*, drew from, and record the names for the simulation.

    *code* becomes what last changed each of them, as for any statement that
    changes a variable: a cell above it then sees the generator moved on by a
    cell below rather than by an edit above it.

    *names* is None when the statement read no generator: nothing is recorded.
    """
    if names is None:
        return
    names = frozenset(names)
    tracking_state.carrier_advances[source_hash] = names
    for name in sorted(names):
        if name in user_ns:
            tracking_state.lineage.record(name, advanced_carrier_lineage(statement_key, name), value=user_ns[name])
            tracking_state.executed_cell_codes[name] = code


def carriers_an_entry_advanced(payload: Any, user_ns: dict[str, Any]) -> frozenset[str] | None:
    """The generators the statement stored in *payload* drew from, or None
    when it read none. Read before the hit restores anything.

    An entry written before the names were recorded has only the generators'
    states after the statement: one counts as drawn from when the live
    generator is somewhere else now, which is what the run that wrote the
    entry saw when the key matched the generator's lineage.
    """
    if not isinstance(payload, dict):
        return None
    recorded = payload.get(PAYLOAD_FIELD)
    if recorded is not None:
        return frozenset(recorded)
    states = payload.get("rng_object_states")
    if not states:
        return None
    live = carrier_positions(states, user_ns)
    moved = set()
    for name, entry in states.items():
        if name not in live:
            continue
        obj, _now = live[name]
        # Compared through the same helper a run uses: "moved" from the
        # stored post-state to where the generator stands now.
        if moved_carrier_names({name: (obj, entry.get("state"))}, user_ns):
            moved.add(name)
    return frozenset(moved)
