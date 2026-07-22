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
from cash.notebook.randomness import capture_rng_state
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


def test_rewinds_to_own_pre_state_when_no_upstream_anchor():
    checker, state = _checker()

    # Record the drawing cell's start position, as the cell executor does.
    random.seed(1234)
    state.rng_pre_states[_digest(DRAW)] = capture_rng_state()
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


def test_upstream_anchor_still_wins_when_present():
    """The pre-state is a FALLBACK; a recorded predecessor keeps priority."""
    checker, state = _checker()

    # An upstream cell that touched RNG, with a recorded post-state.
    random.seed(99)
    state.observed_rng_cells[_digest(CELLS[0])] = {"random"}
    state.rng_post_states[_digest(CELLS[0])] = capture_rng_state()
    from_anchor = random.random()

    # A DIFFERENT own-pre-state, which must not be the one used.
    random.seed(4321)
    state.rng_pre_states[_digest(DRAW)] = capture_rng_state()
    from_own = random.random()
    assert from_anchor != from_own, "test setup must distinguish the two sources"

    random.random()  # move the live stream off both
    checker._restore_position_rng_state(DRAW, CELLS, 1)
    assert random.random() == from_anchor, (
        "an upstream cell with a recorded post-state must remain the rewind anchor"
    )


def test_no_recorded_state_anywhere_is_a_noop():
    """Nothing recorded -> leave the stream alone rather than guess."""
    checker, state = _checker()
    random.seed(7)
    before = capture_rng_state()

    checker._restore_position_rng_state(DRAW, CELLS, 1)

    assert capture_rng_state()["random"] == before["random"]
