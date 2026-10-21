"""The share check reads only the object-column cells something besides the
column holds (`_cells_held_elsewhere`): a million cells only the frame
holds are none, and listing them all cost seconds per check. The C kernel
answers as the Python loop does, a cell that occurs twice in the column
included."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from cash.notebook import shared_objects


def _column(seed):
    rng = random.Random(seed)
    kept = [[i] for i in range(20)]
    pool = [*kept, *([100 + i] for i in range(5))]
    cells = []
    for _ in range(300):
        roll = rng.random()
        if roll < 0.5:
            cells.append(rng.choice(pool))
        elif roll < 0.7:
            cells.append([9])
        else:
            cells.append(rng.choice([None, 1, "s", 2.5, True, b"x", 10**30]))
    return kept, pd.DataFrame({"a": cells, "n": range(300)})


@pytest.mark.parametrize("seed", range(10))
def test_c_and_python_list_the_same_cells(seed, monkeypatch):
    kept, frame = _column(seed)
    kernel = shared_objects._CELLS_HELD_ELSEWHERE
    if kernel is None:
        pytest.skip("no C kernels in this build")
    by_c = shared_objects._object_cells(frame)
    monkeypatch.setattr(shared_objects, "_CELLS_HELD_ELSEWHERE", None)
    by_python = shared_objects._object_cells(frame)
    assert [id(cell) for cell in by_c] == [id(cell) for cell in by_python]
    assert {id(cell) for cell in by_python} == {id(cell) for cell in kept if any(c is cell for c in frame["a"])}


@pytest.mark.parametrize("kernel", ["c", "python"])
def test_cells_only_the_frame_holds_are_not_listed(kernel, monkeypatch):
    if kernel == "python":
        monkeypatch.setattr(shared_objects, "_CELLS_HELD_ELSEWHERE", None)
    elif shared_objects._CELLS_HELD_ELSEWHERE is None:
        pytest.skip("no C kernels in this build")
    cells = [[i] for i in range(10_000)]
    cells += [cells[-1]]  # one cell twice in the column, held by nothing else
    frame = pd.DataFrame({"a": cells})
    del cells
    assert shared_objects._object_cells(frame) == []
    held = frame["a"].iloc[5]
    assert shared_objects._object_cells(frame) == [held]
