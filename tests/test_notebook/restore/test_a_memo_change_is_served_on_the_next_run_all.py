"""A statement whose run changed a ``functools.cache`` memo is served on the
next Run All of the same kernel, as after a restart, and a hit that cannot be
served says why.

Shaped like the TSP notebook's benchmark cells. ``benchmark`` memoises on
the solver, and one solver takes others as defaults
(``ensemble(c, tsps=(greedy, rep))``). ``held_karp`` empties
``shortest_segment`` for each new problem, and the cell above it left that
memo bigger than cash reads. Both ran on every pass: the copy the memory tier
kept held the solvers the first pass defined, which a Run All defined again,
and the memo it empties held more than cash reads.
"""

from __future__ import annotations

from cash.notebook import helper_memos
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LOG = "memo_calls.log"
IMPORTS = f"import functools, os, time\nLOG = {LOG!r}"
NOTE = (
    "def note(tag):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (tag + '\\n').encode())\n"
    "    os.close(fd)"
)
SLEEP = ABOVE_PERSISTENCE_FLOOR_S
SOLVERS = "def greedy(c):\n    return sorted(c)\n\ndef ensemble(c, tsps=(greedy,)):\n    return min(t(c) for t in tsps)"
BENCH = "@functools.cache\ndef bench(tsp, n):\n    note('bench')\n    return [tsp(range(n))]"
SEGMENT = "@functools.cache\ndef seg(n):\n    note('seg')\n    return [n]"
SOLVE = "def solve(n):\n    seg.cache_clear()\n    return [seg(i) for i in range(n)]"
EXACT = "exact = (solve,)"


def _calls(tag: str) -> int:
    try:
        with open(LOG, encoding="utf-8") as fh:
            return fh.read().split().count(tag)
    except FileNotFoundError:
        return 0


def _last(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def _passes(magics, cells: list[str], times: int = 2) -> list[list[str]]:
    """Run *cells* in order *times*; the statuses of the last cell each time."""
    seen = []
    for _ in range(times):
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        seen.append(_last(magics))
    return seen


def _fill_seg(n: int) -> str:
    """A cell leaving ``seg`` with *n* answers, more than cash reads."""
    return f"seg.cache_clear()\nfilled = [seg(i) for i in range(100, 100 + {n})]"


def test_a_memo_keyed_on_a_solver_with_solver_defaults_is_served_on_the_next_run_all(
    cash_magics, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cells = [IMPORTS, NOTE, SOLVERS, BENCH, f"r = (bench(ensemble, 3), time.sleep({SLEEP}))[0]"]
    seen = _passes(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert seen[1] == ["RESTORED"], seen
    assert _calls("bench") == 1
    # The key is the live solver, and the answer the output: a call finds it.
    assert ns["bench"](ns["ensemble"], 3) is ns["r"]
    assert _calls("bench") == 1


def test_a_memo_emptied_before_any_call_is_served_though_it_held_more_than_cash_reads(
    cash_magics, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(helper_memos, "MAX_KEYED", 8)
    cells = [
        IMPORTS,
        NOTE,
        SEGMENT,
        SOLVE,
        EXACT,
        _fill_seg(20),
        f"r = ([t(3) for t in exact], time.sleep({SLEEP}))[0]",
    ]
    seen = _passes(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert seen[1] == ["RESTORED"], seen
    # Emptied, then the run's three answers, as after a run.
    assert ns["seg"].cache_info().currsize == 3
    calls = _calls("seg")
    assert ns["seg"](2) is ns["r"][0][2]
    assert _calls("seg") == calls


def test_a_memo_called_before_it_is_emptied_runs_when_it_held_more_than_cash_reads(cash_magics, tmp_path, monkeypatch):
    """Control: the run found ``seg(100)`` before it emptied the memo, and
    cash did not read what the memo held, so the statement runs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(helper_memos, "MAX_KEYED", 8)
    cells = [
        IMPORTS,
        NOTE,
        SEGMENT,
        SOLVE,
        EXACT,
        _fill_seg(20),
        f"r = ([seg(100)] + [t(3) for t in exact], time.sleep({SLEEP}))[0]",
    ]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen


def test_a_clear_cash_did_not_see_first_is_not_taken_for_the_first(cash_magics, tmp_path, monkeypatch):
    """Control: ``clear``, bound before the statement, empties the memo
    unseen after a call of it; the clear cash sees comes after, and the
    statement runs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(helper_memos, "MAX_KEYED", 8)
    cells = [
        IMPORTS,
        NOTE,
        SEGMENT,
        SOLVE,
        EXACT,
        _fill_seg(20) + "\nclear = seg.cache_clear",
        f"r = ([seg(100), clear()] + [t(3) for t in exact], time.sleep({SLEEP}))[0]",
    ]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen


def test_a_hit_that_cannot_put_a_memo_back_says_why(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = [IMPORTS, NOTE, SOLVERS, BENCH, f"r = (bench(greedy, 3), time.sleep({SLEEP}))[0]"]
    _passes(cash_magics, cells, times=1)
    monkeypatch.setattr(
        helper_memos, "put_back", lambda *a, **k: "the helper memo bench is not one cash can put answers into"
    )
    captured: list[dict] = []
    real_render = cash_magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    monkeypatch.setattr(cash_magics.badges, "render", capture)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    last = captured[-1][-1]
    assert str(last["status"]) == "COMPUTED", last
    assert last.get("miss_reason") == "the helper memo bench is not one cash can put answers into", last
