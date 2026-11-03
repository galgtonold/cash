"""A statement whose run fills or empties a ``functools.cache`` memo is
stored with what it put into the memo, and a hit puts that back.

Plain Run All is the reference: after a hit the memo holds what the run
left, a later call finds its answer without running the helper's body,
and an output that is a memo's answer is the very object the memo holds.
What a hit cannot give back still runs every time: a bounded
``lru_cache`` (its order lives in C), a statement reading ``cache_info``
(the counters), and an answer something outside the entry holds too.
"""

from __future__ import annotations

import warnings

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


def _calls() -> int:
    try:
        with open(LOG, encoding="utf-8") as fh:
            return len(fh.read().split())
    except FileNotFoundError:
        return 0


def _statuses(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def _passes(magics, cells: list[str], times: int = 2) -> list[list[str]]:
    """Run *cells* in order *times*; the statuses of the last cell each time."""
    seen = []
    for _ in range(times):
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        seen.append(_statuses(magics))
    return seen


def _memo(decorator: str = "@functools.cache") -> str:
    return f"{decorator}\ndef f(n):\n    note('f')\n    return [n, n * n]"


def test_a_filled_memo_is_served_and_filled_again(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = [IMPORTS, NOTE, _memo(), f"r = ([f(1), f(2)], time.sleep({SLEEP}))[0]"]
    seen = _passes(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert seen[1] == ["RESTORED"], seen
    assert _calls() == 2  # the second pass ran no body: served, and the memo filled
    assert ns["f"].cache_info().currsize == 2
    assert ns["r"][0] is ns["f"](1)  # the output is the memo's answer, as in a run
    assert _calls() == 2


def test_an_emptied_memo_is_emptied_and_filled_again(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    refill = "def refill():\n    f.cache_clear()\n    time.sleep(%s)\n    return [f(i) for i in range(3)]" % SLEEP
    cells = [IMPORTS, NOTE, _memo(), refill, "seven = f(7)", "r = refill()"]
    seen = _passes(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert seen[1] == ["RESTORED"], seen
    # Emptied, then the run's three answers: the key 7 a cell above put in is gone.
    assert ns["f"].cache_info().currsize == 3
    calls = _calls()
    assert ns["r"][2] is ns["f"](2)
    ns["f"](7)
    assert _calls() == calls + 1


def test_a_bounded_memo_filled_runs_every_time(cash_magics, tmp_path, monkeypatch):
    """Control: a bounded memo keeps its order where only C reaches it."""
    monkeypatch.chdir(tmp_path)
    cells = [IMPORTS, NOTE, _memo("@functools.lru_cache(maxsize=8)"), f"r = ([f(1), f(2)], time.sleep({SLEEP}))[0]"]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen
    assert _calls() == 4


def test_a_statement_reading_the_counters_runs_every_time(cash_magics, tmp_path, monkeypatch):
    """Control: no hit gives back ``cache_info()``'s counts."""
    monkeypatch.chdir(tmp_path)
    cells = [IMPORTS, NOTE, _memo(), f"r = ([f(1), f(2), f.cache_info().misses], time.sleep({SLEEP}))[0]"]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen
    assert cash_magics.shell.user_ns["r"][2] == 2


def test_an_answer_held_outside_the_entry_runs_every_time(cash_magics, tmp_path, monkeypatch):
    """Control: the memo's answer is a variable's list; a hit would put a
    copy into the memo, and a change through the variable would not reach it."""
    monkeypatch.chdir(tmp_path)
    memo = "@functools.cache\ndef f(n):\n    note('f')\n    return L"
    cells = [IMPORTS, NOTE, "L = [1, 2]", memo, f"r = (len(f(1)), time.sleep({SLEEP}))[0]"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen
    assert any("NOTEBOOK-MEMO-SHARED" in str(w.message) for w in caught), [str(w.message) for w in caught]
    assert cash_magics.shell.user_ns["f"](1) is cash_magics.shell.user_ns["L"]


STAMP = "@functools.cache\ndef g(n):\n    note('g')\n    return [n, time.perf_counter()]"


def _two_passes(magics, below: str) -> list[list[str]]:
    """Run the cells twice, the cell above edited on the second pass so it
    runs again and ``g(1)`` gets a different answer; the statuses of the
    statement below each time."""
    seen = []
    for edit in ("", " + []"):
        cells = [IMPORTS, NOTE, STAMP, f"a = g(1){edit}", below]
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        seen.append(_statuses(magics))
    return seen


def test_a_statement_that_found_no_answer_is_served_when_the_answers_above_differ(cash_magics, tmp_path, monkeypatch):
    """``g`` answers hold a timing, so an answer computed again above is a
    different one. The statement below found none of them (it only added
    its own key), so they decide nothing: it is served, and the memo holds
    the answer above and its own, as after a run."""
    monkeypatch.chdir(tmp_path)
    seen = _two_passes(cash_magics, f"r = (g(2), time.sleep({SLEEP}))[0]")
    ns = cash_magics.shell.user_ns
    assert seen[1] == ["RESTORED"], seen
    assert ns["g"].cache_info().currsize == 2
    assert ns["g"](2) is ns["r"]
    assert ns["g"](1)[1] == ns["a"][1]


def test_a_statement_that_found_an_answer_that_differs_now_runs_again(cash_magics, tmp_path, monkeypatch):
    """Control: the statement found ``g(1)`` in the memo, and that answer is
    a different one now, so a run gives another output: it runs."""
    monkeypatch.chdir(tmp_path)
    seen = _two_passes(cash_magics, f"r = (g(2), g(1), time.sleep({SLEEP}))[0:2]")
    ns = cash_magics.shell.user_ns
    assert seen[1] == ["COMPUTED"], seen
    assert ns["r"][1] is ns["g"](1)


def test_a_statement_whose_helper_call_raised_runs_every_time(cash_magics, tmp_path, monkeypatch):
    """A call that raised missed and kept no answer: no check could tell a
    cell above that put that key in, where a run would find it and not
    raise. So the statement is not stored and runs every time."""
    monkeypatch.chdir(tmp_path)
    risky = "@functools.cache\ndef h(n):\n    note('h')\n    if n < 0:\n        raise ValueError(n)\n    return [n]"
    attempt = "def attempt(n):\n    try:\n        return h(n)\n    except ValueError:\n        return None"
    cells = [IMPORTS, NOTE, risky, attempt, f"r = (attempt(-1), attempt(3), time.sleep({SLEEP}))[0:2]"]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen
