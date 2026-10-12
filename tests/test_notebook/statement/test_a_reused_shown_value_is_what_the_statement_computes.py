"""A value a cell just showed is handed to a later statement only where
computing it again would give the same thing.

The shown value of ``f(df)`` was kept by the expression's text and the
lineages of ``f`` and ``df`` alone: ``f`` reading a global that then
changed, ``del df``, ``f`` returning ``df`` itself or a view of it, a column
of lists that a copy shares, and what ``f`` printed were all lost on reuse.
"""

import os

import pytest

from tests._cell_driver import run_cash_cell

pytest.importorskip("pandas")

#: Over the memo's half-second floor, so the shown value is kept.
SLOW = 0.6


def _setup(log) -> str:
    return (
        "import os, time\n"
        "import numpy as np, pandas as pd\n"
        f"LOG = {str(log)!r}\n"
        "def tick():\n"
        "    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        f"    time.sleep({SLOW})\n"
        "K = 2\n"
        "def f(d):\n"
        "    tick()\n"
        "    return d * K\n"
    )


def _runs(log) -> int:
    return log.read_text(encoding="utf-8").count("x") if os.path.exists(log) else 0


DATA = "df = pd.DataFrame({'x': np.arange(5.0)})"


def _run(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def test_the_shown_value_is_reused(cash_magics, mock_shell, tmp_path):
    """Positive control: nothing changed, so the shown value is handed on."""
    log = tmp_path / "runs.log"
    _run(cash_magics, [_setup(log), DATA, "f(df)", "y = f(df)"])
    assert mock_shell.user_ns["y"].x.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert _runs(log) == 1


def test_a_global_the_function_reads_changed_computes_it_again(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "runs.log"
    _run(cash_magics, [_setup(log), DATA, "f(df)", "K = 3", "y = f(df)"])
    assert mock_shell.user_ns["y"].x.tolist() == [0.0, 3.0, 6.0, 9.0, 12.0]
    assert _runs(log) == 2


def test_a_deleted_name_is_not_answered(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "runs.log"
    with pytest.raises(NameError):
        _run(cash_magics, [_setup(log), DATA, "f(df)", "del df", "y = f(df)"])
    assert "y" not in mock_shell.user_ns


def test_a_view_the_function_returns_still_writes_through(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "runs.log"
    setup = _setup(log) + "def middle(g):\n    tick()\n    return g[1:3]\n"
    _run(cash_magics, [setup, "grid = np.ones((4, 4))", "middle(grid)", "m = middle(grid)\nm[:] = 0"])
    assert mock_shell.user_ns["grid"].sum() == 8.0


def test_the_input_the_function_returns_still_writes_through(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "runs.log"
    setup = _setup(log) + "def check(d):\n    tick()\n    return d\n"
    _run(cash_magics, [setup, DATA, "check(df)", "clean = check(df)\nclean['y'] = 1"])
    assert list(mock_shell.user_ns["df"].columns) == ["x", "y"]


def test_a_list_changed_through_one_statement_is_not_seen_by_the_next(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "runs.log"
    setup = _setup(log) + "def split(d):\n    tick()\n    return d.s.str.split(',')\n"
    _run(
        cash_magics,
        [
            setup,
            "df = pd.DataFrame({'s': ['a,b'] * 3})",
            "split(df)",
            "first = split(df)\nfirst.iloc[0].append('X')",
            "again = split(df)",
        ],
    )
    assert mock_shell.user_ns["again"].iloc[0] == ["a", "b"]


def test_what_the_function_printed_is_printed_again(cash_magics, mock_shell, tmp_path, capsys):
    log = tmp_path / "runs.log"
    setup = _setup(log) + "def build(d):\n    print('building report')\n    tick()\n    return d * 2\n"
    cells = [setup, DATA, "build(df)", "res = build(df)"]
    for cell in cells[:3]:
        run_cash_cell(cash_magics, cell, cells=cells)
    capsys.readouterr()
    run_cash_cell(cash_magics, cells[3], cells=cells)
    assert "building report" in capsys.readouterr().out
