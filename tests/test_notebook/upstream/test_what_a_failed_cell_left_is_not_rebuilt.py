"""A cell below a cell that raised part way runs on what that cell left.

``tot = {}`` then a loop filling ``tot`` raises half way. Running a cell
further down, the upstream check re-ran ``tot = {}`` (the lines before the
error) and the cell between (``final = {... tot ...}``) on that state,
silently: tot came back empty and final with it, a state neither a plain
kernel (the partial tot, the previous final) nor a run from the top (it
raises) has. The names the failed run left keep their values, with a
warning, and what was built from them is not rebuilt or restored either.

The same holds when the dict is filled by a function rather than in the
cell's own text: a module helper (``proc.fill(tot, by_user)``) or a
function of the notebook's called in the loop (``put(u)``).
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

PROC = """\
LIMIT = {limit}
def weight(s):
    if s > LIMIT:
        raise ValueError(s)
    return s ** 0.5
def fill(tot, by_user):
    for u in by_user:
        tot[u] = sum(weight(s) for s in by_user[u])
"""

#: The cell that fills ``tot``: in its own text, through a module helper, or
#: through a function of the notebook's.
FILLS = {
    "inline": "tot = {}\nfor u in by_user:\n    tot[u] = sum(cb8proc.weight(s) for s in by_user[u])",
    "helper": "tot = {}\ncb8proc.fill(tot, by_user)",
    "cell_function": (
        "tot = {}\ndef put(u):\n    tot[u] = sum(cb8proc.weight(s) for s in by_user[u])\nfor u in by_user:\n    put(u)"
    ),
}

CELLS = [
    "import cb8proc\nby_user = {'a': [1, 4], 'b': [9, 16], 'c': [25]}",
    "tot = {}\nfor u in by_user:\n    tot[u] = sum(cb8proc.weight(s) for s in by_user[u])",
    "final = {k: v for k, v in tot.items() if v > 2}",
    "seen = (len(final) if 'final' in dir() else None, dict(tot))",
]


@pytest.fixture
def proc(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "cb8proc.py"

    def write(limit: int) -> None:
        path.write_text(PROC.format(limit=limit), encoding="utf-8")
        import importlib
        import sys

        sys.modules.pop("cb8proc", None)
        importlib.invalidate_caches()

    return write


def _run(magics, n, fill="inline"):
    cells = [*CELLS[:1], FILLS[fill], *CELLS[2:]]
    run_cash_cell(magics, cells[n], cells=cells)


@pytest.mark.parametrize("fill", list(FILLS))
def test_a_cell_below_a_failed_loop_sees_what_the_loop_left(cash_magics, mock_shell, proc, fill):
    proc(99)
    for n in range(4):
        _run(cash_magics, n, fill)
    assert mock_shell.user_ns["seen"][0] == 3

    proc(15)
    _run(cash_magics, 0, fill)
    with pytest.raises(ValueError):
        _run(cash_magics, 1, fill)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cash_magics, 3, fill)

    # a plain kernel: tot as the loop left it, final from the run before
    assert mock_shell.user_ns["seen"] == (3, {"a": 3.0})
    assert any(getattr(w.message, "code", None) == "NOTEBOOK-FAILED-CELL" for w in caught), [
        str(w.message) for w in caught
    ]


@pytest.mark.parametrize("fill", list(FILLS))
def test_after_a_restart_what_was_built_from_it_is_not_restored(tmp_path, proc, fill):
    """After a restart, Run All stops at the cell that raises. The next cell
    reads ``final``, which the run before the restart stored: built from a
    whole ``tot``, it is not what this kernel's ``tot`` gives, and a plain
    kernel does not have it."""
    from cash.core import Cash
    from cash.notebook.ipython.magics import CashMagics
    from tests.conftest import MockShell

    def kernel():
        magics = CashMagics(MockShell(), Cash(cache_dir=str(tmp_path / "cache"), register_magic=False))
        magics.cash_on("")
        magics.cash_persist("on")
        magics.badges.mode = "off"
        return magics

    proc(99)
    first = kernel()
    for n in range(4):
        _run(first, n, fill)
    assert first.shell.user_ns["seen"][0] == 3

    proc(15)
    second = kernel()
    _run(second, 0, fill)
    with pytest.raises(ValueError):
        _run(second, 1, fill)
    _run(second, 3, fill)
    assert second.shell.user_ns["seen"] == (None, {"a": 3.0})
    assert "final" not in second.shell.user_ns


def test_a_cell_that_rebinds_the_name_lets_it_be_rebuilt_again(cash_magics, mock_shell, proc):
    """Below a cell that binds ``tot`` anew, ``tot`` is not what the failed
    run left: the cells from there rebuild as usual (passes without the fix
    too; it checks the exclusion stops where it should)."""
    cells = CELLS[:3] + ["tot = {'z': 9.0}", "final = dict(tot)", "seen = final"]
    proc(15)
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with pytest.raises(ValueError):
        run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[5], cells=cells)
    assert mock_shell.user_ns["seen"] == {"z": 9.0}
