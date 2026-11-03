"""A statement that empties a ``functools.cache`` memo before any call of it,
then fills it with more answers than cash checks (`MAX_KEYED`), is served,
and the memo after the hit is the memo after a run: the same keys, equal
answers, an answer that is also an output one object with it, and a later
call that finds its answer without running the body.

Shaped like the TSP notebook's Held-Karp cell, which clears
``shortest_segment`` for each problem and leaves 245,760 answers for 16
cities. Only such a memo is let past the limit: one the statement filled
without emptying it first still runs, and so does one whose keys do not
come back from a pickle equal.
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
#: More answers than cash checks in a memo it did not see emptied.
N = helper_memos.MAX_KEYED + 904
SEGMENT = "@functools.cache\ndef seg(i, part=frozenset()):\n    note('seg')\n    return [i, *sorted(part)]"
SOLVE = (
    "def solve(n, **kw):\n"
    "    seg.cache_clear()\n"
    "    return min((seg(i, frozenset({i % 7, i % 11}), **kw) for i in range(n)), key=lambda s: -s[0])"
)
FILL = "def fill(n):\n    return min((seg(i, frozenset({i % 7})) for i in range(n)), key=lambda s: -s[0])"
SOLVERS = "exact = (solve,)"
#: Leaves ``seg`` holding a few answers, so the clear below is seen as the first.
ABOVE = "small = solve(3)"


def _calls(tag: str) -> int:
    try:
        with open(LOG, encoding="utf-8") as fh:
            return fh.read().split().count(tag)
    except FileNotFoundError:
        return 0


def _last(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def _passes(magics, cells: list[str], times: int = 2) -> list[list[str]]:
    """Run *cells* in order *times*, as Run All does; the statuses of the
    last cell each time."""
    seen = []
    for _ in range(times):
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        seen.append(_last(magics))
    return seen


def _plain(cells: list[str]) -> dict:
    """What a Run All in plain Python leaves in ``seg``: each key's answer."""
    ns: dict = {"__name__": "__plain__"}
    for cell in cells:
        exec(cell.replace(f"LOG = {LOG!r}", "LOG = os.devnull"), ns)
    found = {}
    for i in range(N):
        part = frozenset({i % 7, i % 11})
        found[i] = ns["seg"](i, part)
    # Emptied before its N answers: its counters count them alone.
    assert ns["seg"].cache_info().misses == N and ns["seg"].cache_info().currsize == N
    return {"answers": found, "r": ns["r"], "size": ns["seg"].cache_info().currsize}


def test_a_cleared_memo_with_more_answers_than_cash_checks_is_served_and_equals_plain(
    cash_magics, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    statement = f"r = ([t({N}) for t in exact], time.sleep({SLEEP}))[0][0]"
    cells = [IMPORTS, NOTE, SEGMENT, SOLVE, SOLVERS, ABOVE, statement]
    plain = _plain(cells)
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["RESTORED"]], seen
    ns = cash_magics.shell.user_ns
    assert ns["r"] == plain["r"]
    assert ns["seg"].cache_info().currsize == plain["size"]
    calls = _calls("seg")
    for i in range(N):
        assert ns["seg"](i, frozenset({i % 7, i % 11})) == plain["answers"][i], i
    # Every answer was in the memo: no body ran.
    assert _calls("seg") == calls
    # The output is the memo's answer itself, as after a run.
    assert ns["seg"](N - 1, frozenset({(N - 1) % 7, (N - 1) % 11})) is ns["r"]


def test_a_cleared_memo_called_with_keywords_is_served_with_the_live_keyword_marker(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    statement = f"r = ([t({N}, part=frozenset({{1}})) for t in exact], time.sleep({SLEEP}))[0][0]"
    keyword = SOLVE.replace("seg(i, frozenset({i % 7, i % 11}), **kw)", "seg(i, **kw)")
    cells = [IMPORTS, NOTE, SEGMENT, keyword, SOLVERS, ABOVE, statement]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["RESTORED"]], seen
    ns = cash_magics.shell.user_ns
    calls = _calls("seg")
    # The keys hold functools' marker for the keyword: found by a call.
    assert ns["seg"](5, part=frozenset({1})) == [5, 1]
    assert ns["seg"](N - 1, part=frozenset({1})) is ns["r"]
    assert _calls("seg") == calls


def test_a_memo_filled_past_the_answers_cash_checks_without_a_clear_still_runs(cash_magics, tmp_path, monkeypatch):
    """Control: ``fill`` adds its answers to what the memo held, which cash
    did not read past the limit, so the statement runs every time."""
    monkeypatch.chdir(tmp_path)
    statement = f"r = ([t({N}) for t in filling], time.sleep({SLEEP}))[0][0]"
    cells = [IMPORTS, NOTE, SEGMENT, SOLVE + "\n" + FILL, SOLVERS + "\nfilling = (fill,)", ABOVE, statement]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen


def test_a_cleared_memo_whose_key_does_not_come_back_equal_reruns(cash_magics, tmp_path, monkeypatch):
    """Control: a key holding NaN, which no copy of it equals, so a hit
    could not put back a key a call finds: the statement runs every time."""
    monkeypatch.chdir(tmp_path)
    nan = SOLVE.replace("    return min(", "    seg(float('nan'))\n    return min(")
    statement = f"r = ([t({N}) for t in exact], time.sleep({SLEEP}))[0][0]"
    cells = [IMPORTS, NOTE, SEGMENT, nan, SOLVERS, ABOVE, statement]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen
