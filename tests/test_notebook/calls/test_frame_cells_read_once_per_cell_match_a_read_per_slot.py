"""Fifty frames a hit put back as ``df.iloc[i:]`` hold ``df``'s cells each in
slots of their own. The share check reads each cell once, with a reference
allowed per slot holding it (`_held_beyond_copies`), and finds what a read
of every slot finds: the cells something else holds too, or that hold more
than values. A cell two arrays hold that copy nothing of each other may be
listed besides, never one left out. A result column in an argument column's own memory is linked
row for row unsearched (`cell_links`), as the search links it."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from cash.notebook import frame_cells, shared_objects


class Thing:
    pass


def _every_slot(arrays):
    cells = shared_objects._distinct_slots(arrays)
    found = {id(cell) for cell in shared_objects._cells_held_elsewhere(cells, 2, shared_objects.exact_value_types())}
    del cells
    return found


def _copy(array):
    out = np.empty(len(array), dtype=object)
    out[:] = array
    return out


def _arrays(seed):
    rng = random.Random(seed)
    base = np.empty(400, dtype=object)
    base[:] = [{"i": i} if i % 7 else [Thing()] if i % 2 else [i] for i in range(400)]
    if seed % 2:
        base[5] = base[9]  # one cell in two rows
    arrays = [base]
    for _ in range(rng.randrange(2, 12)):
        start = rng.randrange(0, 50)
        roll = rng.random()
        if roll < 0.5:
            arrays.append(_copy(base[start:]))
        elif roll < 0.8:
            arrays.append(_copy(base[sorted(rng.sample(range(400), 60))]))
        else:
            other = np.empty(30, dtype=object)
            other[:] = [base[rng.randrange(400)] if rng.random() < 0.5 else {"j": j} for j in range(30)]
            arrays.append(other)
    kept = [base[rng.randrange(400)] for _ in range(rng.randrange(0, 4))]
    return arrays, kept


@pytest.mark.parametrize("seed", range(16))
def test_reading_each_cell_once_leaves_out_nothing_reading_each_slot_finds(seed):
    arrays, kept = _arrays(seed)
    once = shared_objects._held_beyond_copies(arrays)
    if once is None:
        pytest.skip("no copies among these arrays")
    assert once >= _every_slot(arrays)
    assert {id(cell) for cell in kept} <= once


@pytest.mark.parametrize("seed", range(8))
def test_copies_and_filters_alone_are_read_exactly(seed):
    rng = random.Random(seed)
    base = np.empty(300, dtype=object)
    base[:] = [{"i": i} if i % 5 else [Thing()] for i in range(300)]
    arrays = [base, *(_copy(base[rng.randrange(40) :]) for _ in range(6))]
    arrays += [_copy(base[sorted(rng.sample(range(300), 50))]) for _ in range(3)]
    kept = [base[rng.randrange(300)] for _ in range(2)]
    assert shared_objects._held_beyond_copies(arrays) == _every_slot(arrays)
    assert {id(cell) for cell in kept} <= _every_slot(arrays)


def test_a_cell_held_outside_every_copy_is_still_found():
    base = np.empty(1000, dtype=object)
    base[:] = [{"i": i} for i in range(1000)]
    arrays = [base, *(_copy(base[i % 5 :]) for i in range(50))]
    assert shared_objects._held_beyond_copies(arrays) == set()
    kept = base[3]
    assert shared_objects._held_beyond_copies(arrays) == {id(kept)} == _every_slot(arrays)


@pytest.mark.parametrize("values", [[], [5], [3, 3, 1], list(range(1000, 0, -3)) * 2])
def test_the_unique_addresses_are_numpys(values):
    array = np.asarray(values, dtype=np.int64)
    assert np.array_equal(frame_cells.sorted_unique(array), np.unique(array))
    probe = np.asarray([0, 1, 3, 5, 7, 999, 1000, 2000], dtype=np.int64)
    assert np.array_equal(frame_cells.in_sorted(probe, np.unique(array)), np.isin(probe, array))


@pytest.mark.parametrize("start", [0, 1, 7])
def test_a_view_of_the_argument_is_linked_as_the_search_links_it(start, monkeypatch):
    df = pd.DataFrame({"o": [{"i": i} for i in range(50)], "n": range(50)})
    view = df.iloc[start:]
    fast = frame_cells.cell_links(view, (df,), {})
    monkeypatch.setattr(frame_cells, "_inside", lambda array, spans: None)
    searched = frame_cells.cell_links(view, (df,), {})
    assert fast[0] == searched[0]
    assert {key: ids.tolist() for key, ids in fast[1].items()} == {
        key: ids.tolist() for key, ids in searched[1].items()
    }
