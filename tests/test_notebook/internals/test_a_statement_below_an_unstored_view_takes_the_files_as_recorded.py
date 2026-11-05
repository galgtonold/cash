"""A statement below an unstored view takes its files' records from above.

A statement records the files its inputs were built from as the input's
producing entry recorded them. ``X_train = X[:5]`` stores no entry, so every
statement below it (``X_train.shape``) took a stat and a digest of each file
``X`` was read from again, 15,000 of them per trivial cell on a photo set.
They are now taken from the entry of the statement ``X_train`` was made
from (``StatementStore._records_through_inputs``).
"""

from __future__ import annotations

import collections

import pytest

from cash.tracking import file_dep_snapshot
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 30


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "data"
    folder.mkdir()
    for i in range(FILES):
        (folder / f"f{i:02d}.txt").write_text("x" * (i + 1), encoding="utf-8")
    return sorted(str(p) for p in folder.iterdir())


def _entries_holding(state, path: str) -> list[dict]:
    return [snap[path] for (snap,) in state.entry_file_states.values() if snap and path in snap]


def test_a_statement_below_an_unstored_view_takes_no_stat_of_the_files(cash_magics, paths, monkeypatch):
    run_cash_cell(cash_magics, "import numpy as np\nimport time")
    run_cash_cell(
        cash_magics,
        f"X = np.array([len(open(p).read()) for p in {paths!r}])\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})",
    )
    run_cash_cell(cash_magics, "X_train = X[:5]")
    state = cash_magics.tracking_state
    assert state.variable_sources.get("X_train") not in state.entry_file_states, (
        "X_train's statement stored an entry: the case below is not reached"
    )
    assert set(paths) <= state.executed_file_deps["X_train"]

    stats: collections.Counter = collections.Counter()
    real = file_dep_snapshot.own_stat

    def counting(path, *args, **kwargs):
        if str(path) in paths:
            stats["stat"] += 1
        return real(path, *args, **kwargs)

    monkeypatch.setattr(file_dep_snapshot, "own_stat", counting)
    # Stored: over the floor (a trivial one under it is not stored at all).
    run_cash_cell(cash_magics, f"n = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), X_train.shape[0] + 1)[1]")
    monkeypatch.setattr(file_dep_snapshot, "own_stat", real)

    assert stats["stat"] == 0, stats
    for path in paths:
        recorded = _entries_holding(state, path)
        assert len(recorded) == 2, f"{path} is not in both entries"
        assert recorded[0] == recorded[1], "the statement below recorded the file anew"


def test_an_input_rebound_since_is_not_followed(cash_magics, statement_processor, paths):
    """The records are taken from above only while the input the view was
    made from is still that value: once ``X`` is rebound, they are taken anew."""
    run_cash_cell(cash_magics, "import numpy as np\nimport time")
    run_cash_cell(
        cash_magics,
        f"X = np.array([len(open(p).read()) for p in {paths!r}])\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})",
    )
    run_cash_cell(cash_magics, "X_train = X[:5]")
    store = statement_processor._store
    taken = store._records_through_inputs("X_train")
    assert set(paths) <= {path for records in taken for path in records.paths}  # the positive control
    run_cash_cell(cash_magics, "X = np.zeros(3)")
    assert store._records_through_inputs("X_train") == []
