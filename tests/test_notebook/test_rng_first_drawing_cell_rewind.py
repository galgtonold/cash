"""A drawing cell with no upstream RNG cell rewinds to its OWN start (CAS-229).

Reproducing a re-executed draw means rewinding the stream to where the draw
STARTED. cash records each RNG-touching cell's *post*-state, so
``_restore_position_rng_state`` could only rewind by finding some upstream cell
that touched RNG and had been recorded. When the drawing cell is the first to
touch the stream there is no such predecessor, the restore silently did nothing,
and the re-executed draw continued from the live stream and returned a different
value. That matters because a cheap draw is under the persistence floor, so it
is RE-EXECUTED rather than served from cache — the rewind is the only thing
making it reproducible.

The bug was masked in practice: any upstream cell that merely caused cash to
import numpy got recorded as having changed ``numpy.random`` and so became a
usable rewind anchor. A plain ``import random`` cell would qualify. These tests
pin the real behaviour so it cannot regress if that incidental signal goes away
(as it must when the cell- and statement-level RNG observers are merged).
"""
from __future__ import annotations

import hashlib
import random
from unittest.mock import MagicMock

from cash.notebook._protocols import TrackingState
from cash.notebook.randomness import (
    capture_rng_state,
    rng_lineage_fingerprint,
    rng_virtual_var,
)
from cash.notebook.upstream import UpstreamChecker

CELLS = ["import random", "rv = random.random()"]
DRAW = CELLS[1]


def _checker() -> tuple[UpstreamChecker, TrackingState]:
    shell = MagicMock()
    shell.user_ns = {}
    checker = UpstreamChecker(shell, MagicMock(), debug=False)
    state = TrackingState()
    checker.set_tracking_state(state)
    return checker, state


def _digest(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _record_pre(state, cell: str, modules=("random",)) -> None:
    """Store a start position the way the cell executor does."""
    state.rng_pre_states[_digest(cell)] = (
        capture_rng_state(),
        rng_lineage_fingerprint(state.variable_lineage, set(modules)),
    )


def test_rewinds_to_own_pre_state_when_no_upstream_anchor():
    checker, state = _checker()

    random.seed(1234)
    _record_pre(state, DRAW)
    expected = random.random()  # the value a draw from that position yields

    # Leave the live stream somewhere else entirely.
    random.random()
    random.random()

    # No observed_rng_cells and no rng_post_states -> no upstream anchor at all.
    assert not state.observed_rng_cells
    assert not state.rng_post_states

    checker._restore_position_rng_state(DRAW, CELLS, 1)
    assert random.random() == expected, (
        "a first-drawing cell must rewind to the position it started from"
    )


def test_own_pre_state_wins_over_upstream_anchor():
    """The cell's own start is exact; the upstream anchor only approximates it."""
    checker, state = _checker()

    random.seed(99)
    state.observed_rng_cells[_digest(CELLS[0])] = {"random"}
    state.rng_post_states[_digest(CELLS[0])] = capture_rng_state()
    from_anchor = random.random()

    random.seed(4321)
    _record_pre(state, DRAW)
    from_own = random.random()
    assert from_anchor != from_own, "test setup must distinguish the two sources"

    random.random()  # move the live stream off both
    checker._restore_position_rng_state(DRAW, CELLS, 1)
    assert random.random() == from_own, (
        "the cell's own recorded start must take priority over the upstream anchor"
    )


def test_stale_pre_state_is_rejected_when_the_seed_changed():
    """A position recorded under a different seed must not be reused."""
    checker, state = _checker()

    # Upstream anchor available as the fallback.
    random.seed(99)
    state.observed_rng_cells[_digest(CELLS[0])] = {"random"}
    state.rng_post_states[_digest(CELLS[0])] = capture_rng_state()
    from_anchor = random.random()

    # Own position recorded while seed lineage was "seed-v1"...
    state.variable_lineage[rng_virtual_var("random")] = "seed-v1"
    random.seed(4321)
    _record_pre(state, DRAW)

    # ...but the seed has since changed. The saved position belongs to the old
    # seed, so it must be discarded in favour of the upstream anchor.
    state.variable_lineage[rng_virtual_var("random")] = "seed-v2"

    random.random()
    checker._restore_position_rng_state(DRAW, CELLS, 1)
    assert random.random() == from_anchor, (
        "a pre-state recorded under a superseded seed must not be restored"
    )


def test_unseeded_pre_state_stays_valid():
    """No seed anywhere -> the frozen position keeps applying across runs."""
    checker, state = _checker()

    random.seed(2024)
    _record_pre(state, DRAW)
    expected = random.random()

    random.random()
    checker._restore_position_rng_state(DRAW, CELLS, 1)
    first = random.random()
    random.random()
    checker._restore_position_rng_state(DRAW, CELLS, 1)
    second = random.random()

    assert first == expected == second, (
        "an unseeded stream's frozen position must survive repeated rewinds"
    )


def test_no_recorded_state_anywhere_is_a_noop():
    """Nothing recorded -> leave the stream alone rather than guess."""
    checker, state = _checker()
    random.seed(7)
    before = capture_rng_state()

    checker._restore_position_rng_state(DRAW, CELLS, 1)

    assert capture_rng_state()["random"] == before["random"]
