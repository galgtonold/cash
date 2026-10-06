"""``for a, b in rows`` with a row of the wrong size raises, as in plain Python.

Binding a loop target used ``zip(..., strict=False)``, so a row with too many
or too few items was cut or half bound and the loop ran on. Iterating a
DataFrame yields column names, so ``for a, row in df:`` ran the body over the
first two letters of each name instead of raising ``ValueError``.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    ("loop", "message"),
    [
        ("for a, b in [(1, 2, 3)]:\n    pass", "too many values to unpack \\(expected 2\\)"),
        ("for a, b in ['day']:\n    pass", "too many values to unpack \\(expected 2\\)"),
        ("for a, b, c in [(1, 2)]:\n    pass", "not enough values to unpack \\(expected 3, got 2\\)"),
        ("for a, (b, c) in [(1, (2, 3, 4))]:\n    pass", "too many values to unpack \\(expected 2\\)"),
        ("for a, *r, z in [(1,)]:\n    pass", "not enough values to unpack \\(expected at least 2, got 1\\)"),
    ],
)
def test_a_row_of_the_wrong_size_raises(cash_magics, loop, message):
    with pytest.raises(ValueError, match=message):
        run_cash_cell(cash_magics, loop)


def test_a_starred_target_collects_the_rest(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "seen = []\nfor a, *rest, z in [(1, 2, 3, 4), (5, 6)]:\n    seen.append((a, rest, z))")
    assert mock_shell.user_ns["seen"] == [(1, [2, 3], 4), (5, [], 6)]
