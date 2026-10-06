"""A frame holding lists is copied a container at a time, not through pickle.

`df = pd.DataFrame(logs, columns=["user", "actions"])` over 87,000 parsed
sessions has a list of (action, datetime) pairs in every row. Copying that
frame by pickle round trip paid per leaf, on every store and every RAM hit:
2.5 s at 800,000 pairs. The lists are what a caller can write into, so they
are what is copied; the tuples of immutables around the pairs are shared.
"""

from __future__ import annotations

import datetime
import pickle

import pytest

from cash import kept_state
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

pd = pytest.importorskip("pandas")

pytestmark = [pytest.mark.core]

STAMP = datetime.datetime(2019, 9, 20, 13, 44, 16)


def _sessions(n: int = 40) -> "pd.DataFrame":
    rows = [
        (f"@User{i}", [(f"Action_{j % 9}", STAMP + datetime.timedelta(minutes=j)) for j in range(1 + i % 7)])
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=["user", "actions"])


class _KeptStateThatCannotDump:
    """The backend's pickling helper, except that dumping is refused.

    Patching the pickle module itself would break every other user of it, the
    store's size estimate among them.
    """

    def __getattr__(self, name):
        return getattr(kept_state, name)

    def dumps(self, *_a, **_k):
        raise AssertionError("the frame went through a pickle round trip")


@pytest.fixture
def no_pickle(monkeypatch):
    monkeypatch.setattr(memory_backend, "kept_state", _KeptStateThatCannotDump())


def test_a_frame_of_lists_is_stored_and_restored_as_independent_copies():
    b = InMemoryBackend()
    frame = _sessions()
    expected = pickle.loads(pickle.dumps(frame))  # pandas deep copies never copy the cells
    b.set("k", frame)
    first = b.get("k")[1]
    assert first.equals(expected)
    assert first is not b.get("k")[1]
    first.actions.iloc[3].append(("the caller's", None))
    first.at[5, "actions"] = []
    assert b.get("k")[1].equals(expected), "a change by the caller reached the stored frame"
    frame.actions.iloc[2].append(("the original's", None))
    assert b.get("k")[1].equals(expected), "the stored frame shares the original's lists"


def test_the_copy_holds_new_lists_and_the_same_immutable_pairs(no_pickle):
    frame = _sessions()
    copied = InMemoryBackend._copy_frame(frame)
    frame.actions.iloc[1].append(("the original's", None))
    assert len(copied.actions.iloc[1]) == 2
    frame.actions.iloc[1].pop()
    assert copied.equals(frame)
    for old, new in zip(frame.actions, copied.actions):
        assert new is not old
        assert all(a is b for a, b in zip(old, new))


def test_a_column_of_tuples_of_immutables_is_shared_whole(no_pickle):
    frame = _sessions().explode("actions")
    copied = InMemoryBackend._copy_frame(frame)
    assert copied.equals(frame)
    assert all(a is b for a, b in zip(frame.actions, copied.actions))
    copied["extra"] = 1
    assert "extra" not in frame.columns


def test_a_series_of_lists_is_copied_cell_by_cell(no_pickle):
    series = _sessions().actions
    copied = InMemoryBackend._copy_frame(series)
    assert copied.equals(series) and copied.name == series.name and copied.index.equals(series.index)
    copied.iloc[1].append(("x", None))
    assert len(series.iloc[1]) == 2


def test_two_list_columns_are_copied_together():
    frame = pd.DataFrame({"a": [[1], [2, 3]], "b": [[4, 5], [6]], "n": [1, 2]})
    copied = InMemoryBackend._copy_frame(frame)
    assert copied.equals(frame)
    copied.a.iloc[0].append(9)
    copied.b.iloc[1].append(9)
    assert frame.a.iloc[0] == [1] and frame.b.iloc[1] == [6]


def test_a_list_held_by_two_cells_is_still_one_list_after_the_copy():
    shared = [1, 2]
    frame = pd.DataFrame({"a": [shared, [3]], "b": [[4], shared]})
    copied = InMemoryBackend._copy_frame(frame)
    assert copied.a.iloc[0] is copied.b.iloc[1]
    assert copied.a.iloc[0] is not shared


def test_cells_that_are_not_plain_data_still_get_a_copy():
    frame = pd.DataFrame({"d": [{"k": [1]}, {"k": [2]}]})
    copied = InMemoryBackend._copy_frame(frame)
    copied.d.iloc[0]["k"].append(9)
    assert frame.d.iloc[0] == {"k": [1]}


def test_the_copy_equals_what_pickle_would_have_made():
    frame = _sessions(200)
    assert InMemoryBackend._copy_frame(frame).equals(pickle.loads(pickle.dumps(frame)))
