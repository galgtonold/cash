"""Parsed rows holding pandas timestamps and numpy scalars are read a level
at a time by the RAM tier's sizing and by the walk for held notebook
functions, as rows of plain values are.

JuNE expert_2 parses an 87,000-line log into ``general_log``, a list of
``{'user_id': ..., 'action_time': pd.Timestamp(...), ...}``. Every store of
it sized the rows a timestamp at a time (0.7 s per 30,000 rows), and the
statements reading it walked each timestamp for the functions it might
hold. The loop cell ran past the replay's 300 s limit under cash.
"""

from __future__ import annotations

import sys

import pytest

from cash import _plain_data, sizing
from cash.notebook import held_functions

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _rows(n: int) -> list[dict]:
    return [
        {
            "user_id": f"@User{i % 7}",
            "session_id": i,
            "action_time": pd.Timestamp("2019-10-15 18:08:02") + pd.Timedelta(seconds=i),
            "score": np.float64(i) / 2,
            "missing": pd.NaT,
        }
        for i in range(n)
    ]


def test_rows_of_timestamps_are_sized_as_a_tree_and_count_every_leaf():
    rows = _rows(50)
    assert _plain_data.tree_size(rows) is None  # not plain data over the default leaves
    size = sizing.memory_footprint(rows)
    by_object = sys.getsizeof(rows)
    seen: set[int] = set()
    for row in rows:
        by_object += sys.getsizeof(row)
        for key, value in row.items():
            for part in (key, value):
                if id(part) not in seen:
                    seen.add(id(part))
                    by_object += sys.getsizeof(part)
    # The level-at-a-time size samples a level's keys, never fewer than it holds.
    assert size >= by_object * 0.9


def test_the_sizing_walks_no_row_one_at_a_time(monkeypatch):
    visited = []
    real = sizing._own_footprint

    def counting(obj, seen, stack):
        visited.append(type(obj))
        return real(obj, seen, stack)

    monkeypatch.setattr(sizing, "_own_footprint", counting)
    sizing.memory_footprint(_rows(1_000))
    assert visited == [list]


def test_a_notebook_function_beside_timestamps_is_still_found():
    namespace: dict = {}
    exec("def make(x):\n    return x\n", namespace)
    rows = _rows(20)
    rows[7]["hook"] = namespace["make"]
    held = held_functions.held_notebook_code(rows, namespace)
    assert namespace["make"].__code__ in held.codes


def test_rows_of_timestamps_hold_no_function_and_are_not_walked_one_by_one(monkeypatch):
    namespace: dict = {}
    exec("def make(x):\n    return x\n", namespace)
    groups = []
    real = held_functions._visit_group

    def counting(kind, group, *rest):
        groups.append((kind, len(group)))
        return real(kind, group, *rest)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    held = held_functions.held_notebook_code(_rows(1_000), namespace)
    assert held.codes == ()
    assert pd.Timestamp not in {kind for kind, _n in groups}
