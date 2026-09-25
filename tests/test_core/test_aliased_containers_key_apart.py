"""One list held twice keys apart from two equal lists.

``[[0] * 3] * 3`` is one row three times: written into, it changes in three
places. Keyed by content alone it matched a real 3x3 grid, so a user who
fixed the classic aliasing bug was still served the aliased grid's result.
"""

from __future__ import annotations

import copy

import pytest

from cash import Cash, FileBackend


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda *args: c._args.hash_payload(args, {})


def test_a_fixed_grid_gets_its_own_result(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def mark_corner(grid):
        new = copy.deepcopy(grid)
        new[0][0] = 1
        return new

    assert mark_corner([[0] * 3] * 3) == [[1, 0, 0]] * 3
    assert mark_corner([[0] * 3 for _ in range(3)]) == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    "aliased, separate",
    [
        (lambda: [[0] * 3] * 3, lambda: [[0] * 3 for _ in range(3)]),
        (lambda: (lambda row: [row, [row]])([0]), lambda: [[0], [[0]]]),
        (lambda: (lambda b: [b, 1, b])(bytearray(b"x")), lambda: [bytearray(b"x"), 1, bytearray(b"x")]),
        (lambda: [{"a": [1]}] * 2, lambda: [{"a": [1]}, {"a": [1]}]),
        (lambda: (lambda s: [s, s, {1}])({1}), lambda: [{1}, {1}, {1}]),
        (lambda: (lambda row: {"x": row, "y": row})([1]), lambda: {"x": [1], "y": [1]}),
    ],
    ids=["grid rows", "across levels", "bytearray leaves", "dict rows", "sets in a list", "dict values"],
)
def test_an_aliased_value_keys_apart_from_its_copy(key, aliased, separate):
    assert key(aliased()) != key(separate())
    assert key(aliased()) == key(aliased())


def test_the_same_container_as_two_arguments_keys_apart(key):
    row = {"n": [1]}
    assert key(row, row) != key(row, {"n": [1]})
    rows = [[1, 2]]
    assert key(rows, rows) != key(rows, [[1, 2]])


def test_a_row_held_elsewhere_is_not_an_alias(key):
    """Positive control: a variable naming one row adds a reference; the
    rows are still separate and key like a fresh copy."""
    from cash._plain_data import aliases

    rows = [[i, str(i)] for i in range(100)]
    held = rows[7]
    assert aliases(rows) == ()
    assert key(rows) == key([[i, str(i)] for i in range(100)])
    assert held is rows[7]
