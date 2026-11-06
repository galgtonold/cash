"""A snapshot asks whether a statement's share-edge record changed since it.

The snapshot scan asks it of every cell above the checked one, on every
check. It sliced the list of every change since the snapshot's
generation, a list that grows with every statement run;
`TrackingState.share_edges_changed_since` answers from the index of each
statement's last change instead, and must answer as the slice did.
"""

from __future__ import annotations

import random

from cash.notebook.tracking_state import TrackingState


def test_the_index_answers_as_the_changes_since_the_generation():
    rng = random.Random(7)
    state = TrackingState()
    hashes = [f"h{i}" for i in range(6)]
    for _ in range(40):
        state.note_share_edge_change(rng.choice(hashes))
    changes = state.share_edge_changes
    assert len(changes) == 40
    for generation in range(len(changes) + 1):
        for size in (0, 1, 2, 3):
            asked = frozenset(rng.sample(hashes, size))
            assert state.share_edges_changed_since(generation, asked) == (not asked.isdisjoint(changes[generation:])), (
                generation,
                asked,
            )


def test_a_reset_forgets_the_index():
    state = TrackingState()
    state.note_share_edge_change("h")
    state.reset_session_state()

    assert state.share_edge_changes == []
    assert not state.share_edges_changed_since(0, {"h"})
