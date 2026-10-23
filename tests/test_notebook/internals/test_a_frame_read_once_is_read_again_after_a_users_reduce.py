"""Within a statement's checks cash reads each frame's slots once. A
``__reduce__`` of the user's that one of cash's pickles runs in between may
write into those slots: what was read is read again, once, and the digest
taken after it is the slots' new one.
"""

from __future__ import annotations

import numpy as np

from cash import kept_state, value_hash
from cash.notebook import frame_cells


def _cells() -> np.ndarray:
    cells = np.empty(2_000, dtype=object)
    cells[:] = [{"i": i} for i in range(2_000)]
    return cells


class _Writer:
    def __init__(self, cells: np.ndarray) -> None:
        self.cells = cells

    def __reduce__(self):
        self.cells[3] = {"i": 3, "z": 1}
        return (dict, ())


class _Plain:
    pass


def test_a_reduce_of_the_users_reads_the_slots_again():
    cells = _cells()
    with frame_cells.reading_once():
        before = frame_cells.slots_digest(cells)
        kept_state.dumps(_Writer(cells))
        assert frame_cells.slots_digest(cells) != before


def test_without_one_the_slots_are_read_once():
    cells = _cells()
    with frame_cells.reading_once():
        before = frame_cells.slots_digest(cells)
        kept_state.dumps(_Plain())
        cells[3] = {"i": 3, "z": 1}  # cash's own code never does: kept
        assert frame_cells.slots_digest(cells) == before
    assert frame_cells.slots_digest(cells) != before


def test_a_value_read_once_is_read_again_too():
    kept = np.arange(100_000.0)
    with value_hash.one_read():
        before = value_hash.compute_hash(kept)
        kept_state.dumps(_Writer(_cells()))
        kept[0] = -1.0
        assert value_hash.compute_hash(kept) != before
