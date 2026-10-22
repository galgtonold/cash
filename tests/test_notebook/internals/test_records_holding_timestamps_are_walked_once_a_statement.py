"""Records holding a ``pd.Timestamp`` are walked once in a statement's look,
whatever leaves each check takes.

Such records are no tree over the default leaves, so every check asking
with the library's value types among its leaves (``is_tree``,
``held_beyond_parents``) walked them again: five walks of two million
parsed log rows after JuNE expert_2's loop, 3.5 s of each run. The look
keeps a walk over the wider leaves too. A tree it found names every type
it holds, so it answers for any leaves; "no tree" answers only for leaves
within those it was walked over. Outside a look, or in the next one, the
value is read again.
"""

from __future__ import annotations

import pytest

from cash import _plain_data
from cash.notebook.shared_objects import object_free_leaves, tree_leaves

pd = pytest.importorskip("pandas")

if _plain_data._K is None:
    pytest.skip("the C walk is not built", allow_module_level=True)


def _rows(n: int) -> list:
    when = pd.Timestamp("2021-05-03 01:43:11")
    return [{"user": f"@User{i}", "session": i, "time": when} for i in range(n)]


class _Counting:
    def __init__(self, kernels):
        self._kernels = kernels
        self.walks = 0

    def __getattr__(self, name):
        return getattr(self._kernels, name)

    def tree_facts(self, *args):
        self.walks += 1
        return self._kernels.tree_facts(*args)


@pytest.fixture
def kernels(monkeypatch):
    counting = _Counting(_plain_data._K)
    monkeypatch.setattr(_plain_data, "_K", counting)
    return counting


def test_every_check_of_a_look_answers_from_one_wider_walk(kernels):
    rows = _rows(1000)
    with _plain_data.one_look({"rows": rows}):
        answers = [
            _plain_data.is_tree(rows, object_free_leaves()),
            _plain_data.is_tree(rows, tree_leaves()),
            _plain_data.held_beyond_parents(rows, tree_leaves(), 10),
            _plain_data.is_tree(rows, object_free_leaves()),
        ]
    assert answers == [True, True, [], True]
    # The default leaves' walk (no tree: a Timestamp), then one wider walk.
    assert kernels.walks == 2


def test_the_answers_are_the_walks_without_a_look():
    rows = _rows(50)
    rows[3]["time"] = object()
    held = _rows(50)
    keep = held[7]
    for value in (rows, held):
        outside = [
            _plain_data.is_tree(value, object_free_leaves()),
            _plain_data.is_tree(value, tree_leaves()),
            _plain_data.held_beyond_parents(value, tree_leaves(), 10),
        ]
        with _plain_data.one_look({"value": value}):
            inside = [
                _plain_data.is_tree(value, object_free_leaves()),
                _plain_data.is_tree(value, tree_leaves()),
                _plain_data.held_beyond_parents(value, tree_leaves(), 10),
            ]
        assert inside == outside
    assert keep is held[7]


def test_a_value_changed_after_a_look_is_read_again(kernels):
    rows = _rows(100)
    with _plain_data.one_look({"rows": rows}):
        assert _plain_data.is_tree(rows, object_free_leaves())
    rows.append({"user": "x", "session": 0, "time": object()})
    with _plain_data.one_look({"rows": rows}):
        assert not _plain_data.is_tree(rows, object_free_leaves())
    assert not _plain_data.is_tree(rows, object_free_leaves())


def test_a_name_bound_to_another_value_is_not_answered_for_the_first(kernels):
    first, second = _rows(10), _rows(10)
    second[0]["time"] = object()
    named = {"rows": first}
    with _plain_data.one_look(named):
        assert _plain_data.is_tree(first, object_free_leaves())
        named["rows"] = second
        assert not _plain_data.is_tree(second, object_free_leaves())
