"""A statement over a helper that provably computes from its arguments alone
is served where the memo lacks an answer it read, or already holds one it
computed: a run would compute the one, the same, and find the other.

The TSP notebook's ``subsegment_indexes(N)`` returns a list of index pairs,
built from ``range`` and ``reversed`` only. Its benchmark cells ran again
on two passes of the scripted session:

- after a restart: "the helper memo subsegment_indexes does not hold the
  answers the statement read". The cell had run in a warm kernel, where the
  memo held answers the cells below had put there, and the record asks for
  every answer it held (the counters do not say which one it found);
- after an edit above, running the cells below in the same kernel: "the
  helper memo subsegment_indexes already holds an answer the statement
  computed", the very answer the cell's own last hit had put back.

For such a helper (``helper_memos._pure``) a lookup now calls it for an
answer the run read that the memo lacks, and asks for the digest the run
read; and takes a key the run computed that the memo holds when it holds
the answer the run left and the helper's body gives that answer again. The
memo's answer stays, so the entry is not served when its outputs hold the
run's answer: they would hold the memo's own.
"""

from __future__ import annotations

from cash.notebook import helper_memos
from cash.notebook.ipython.session import CashSession
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = ABOVE_PERSISTENCE_FLOOR_S
SEG = (
    "import functools, time\n\n@functools.cache\n"
    "def seg(n):\n    return [(i, i + length) for length in reversed(range(2, n)) for i in range(n - length)]"
)
FILL = f"a = (len(seg(5)) + len(seg(6)), time.sleep({SLEEP}))[0]"
READ = f"b = (len(seg(5)) + len(seg(9)), time.sleep({SLEEP}))[0]"
BELOW = f"c = (len(seg(7)), time.sleep({SLEEP}))[0]"


def _last(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def _run(magics, cells: list[str], order: list[int] | None = None) -> list[list[str]]:
    seen = []
    for i in order if order is not None else range(len(cells)):
        run_cash_cell(magics, cells[i], cells=cells)
        seen.append(_last(magics))
    return seen


def _reasons(monkeypatch) -> dict:
    """``code -> why it ran`` for the statements run from now on: the miss
    reason, or why it could not be stored."""
    reasons: dict = {}
    record = CashSession.record_cell

    def record_cell(self, metrics, *args, **kwargs):
        for m in metrics:
            reasons[str(m.get("code"))] = (m.get("miss_reason"), m.get("uncacheable_reasons"))
        return record(self, metrics, *args, **kwargs)

    monkeypatch.setattr(CashSession, "record_cell", record_cell)
    return reasons


def _restart(magics) -> None:
    """The memo as a kernel restart leaves it: empty."""
    magics.shell.user_ns["seg"].cache_clear()


def _length(n: int) -> int:
    return sum(n - length for length in range(2, n))


# --- a memo lacking an answer the run read ------------------------------------


def test_an_answer_the_run_read_that_the_memo_lacks_is_computed_and_the_entry_served(cash_magics):
    cells = [SEG, FILL, READ, BELOW]
    # The first session ran the cell below first: the memo held seg(7) when
    # READ ran, and its record asks for it.
    _run(cash_magics, cells, [0, 1, 3, 2])
    _restart(cash_magics)
    seen = _run(cash_magics, cells)
    assert seen[2] == ["RESTORED"], seen
    ns = cash_magics.shell.user_ns
    assert ns["b"] == _length(5) + _length(9)
    assert ns["seg"](9) == [(i, i + n) for n in reversed(range(2, 9)) for i in range(9 - n)]


def test_an_answer_the_run_read_changed_in_place_is_not_computed_into_a_hit():
    """Guard: the run read ``seg(5)`` as something had changed it; computed
    again, it is another answer, so the entry is not served. (In a notebook
    the change also moves ``seg``'s lineage: the key differs first.)"""
    ns: dict = {}
    exec(SEG, ns)
    seg = ns["seg"]
    seg(5).append((0, 0))
    before = helper_memos.watched(ns)
    exec("b = len(seg(5)) + len(seg(9))", ns)
    records = helper_memos.used(before, ns=ns, changes=[])
    assert isinstance(records, list) and records[0]["pure"], records
    seg.cache_clear()
    assert helper_memos.unfilled(records, ns) is not None
    # Control: unchanged, the answer it read is computed into the memo.
    exec(SEG, ns)
    ns["seg"](5)
    before = helper_memos.watched(ns)
    exec("b = len(seg(5)) + len(seg(9))", ns)
    records = helper_memos.used(before, ns=ns, changes=[])
    ns["seg"].cache_clear()
    ns["seg"](5)
    assert helper_memos.unfilled(records, ns) is None
    assert ns["seg"].cache_info().currsize == 1


def test_a_helper_reading_a_global_is_not_called_for_an_answer_the_memo_lacks(cash_magics, monkeypatch):
    """Guard: ``seg`` reads ``WIDTH``, which a cell can change: a lookup
    never calls it, and the statement runs."""
    seg = SEG.replace("import functools, time\n", "import functools, time\nWIDTH = 2\n").replace(
        "range(2, n)", "range(WIDTH, n)"
    )
    cells = [seg, FILL, READ, BELOW]
    _run(cash_magics, cells, [0, 1, 3, 2])
    _restart(cash_magics)
    reasons = _reasons(monkeypatch)
    seen = _run(cash_magics, cells)
    assert seen[2] == ["COMPUTED"], (seen, reasons)
    assert "does not hold the answers the statement read" in str(reasons[READ]), reasons


# --- a memo already holding an answer the run computed ------------------------


def test_an_answer_the_run_computed_that_the_memo_holds_is_found_and_the_entry_served(cash_magics, monkeypatch):
    cells = [SEG, FILL, READ]
    _run(cash_magics, cells)
    reasons = _reasons(monkeypatch)
    # The cells below the helper again in the same kernel, as after an edit
    # above them: the memo holds every answer they computed.
    seen = _run(cash_magics, cells, [1, 2])
    assert seen == [["RESTORED"], ["RESTORED"]], (seen, reasons)
    ns = cash_magics.shell.user_ns
    assert (ns["a"], ns["b"]) == (_length(5) + _length(6), _length(5) + _length(9))
    assert ns["seg"].cache_info().currsize == 3


def test_an_answer_the_run_computed_and_changed_in_place_is_not_found_into_a_hit(cash_magics):
    """Guard: the run appended to the answer it computed; a run here finds
    that answer and appends again, so the statement runs."""
    grow = "def grow(n):\n    seg(n).append((0, 0))\n    return n"
    cells = [SEG, grow, f"r = (grow(9), len(seg(9)), time.sleep({SLEEP}))[1]"]
    _run(cash_magics, cells)
    assert _run(cash_magics, cells, [2]) == [["COMPUTED"]]
    assert cash_magics.shell.user_ns["r"] == _length(9) + 2


def test_a_lookup_finding_an_answer_changed_in_place_by_the_run_refuses_it():
    """The same at the lookup: the memo holds the answer as the run left it,
    and the helper computes another (in a notebook the change also moves
    ``seg``'s lineage: the key differs first)."""
    ns: dict = {}
    exec(SEG, ns)
    before = helper_memos.watched(ns)
    exec("seg(9).append((0, 0)); r = len(seg(9))", ns)
    records = helper_memos.used(before, ns=ns, changes=[])
    assert isinstance(records, list) and records[0]["pure"], records
    reason = helper_memos.unfilled(records, ns)
    assert reason is not None and "already holds an answer the statement computed" in reason
    # Control: a run that only read it.
    exec(SEG, ns)
    before = helper_memos.watched(ns)
    exec("r = len(seg(9))", ns)
    records = helper_memos.used(before, ns=ns, changes=[])
    assert helper_memos.unfilled(records, ns) is None


def test_an_answer_the_run_computed_that_its_output_holds_is_not_found_into_a_hit(cash_magics, monkeypatch):
    """Guard: ``x`` is the memo's answer after a run; served, it would be a
    copy, and a change to it would no longer reach the memo."""
    cells = [SEG, f"x = (seg(9), time.sleep({SLEEP}))[0]"]
    _run(cash_magics, cells)
    reasons = _reasons(monkeypatch)
    seen = _run(cash_magics, cells, [1])
    ns = cash_magics.shell.user_ns
    assert seen == [["COMPUTED"]], (seen, reasons)
    assert "already holds an answer the statement computed" in str(reasons[cells[-1]]), reasons
    assert ns["x"] is ns["seg"](9)


def test_a_helper_with_an_effect_is_not_found_into_a_hit(cash_magics, monkeypatch):
    """Guard: ``seg`` reseeds ``random``; a run finding its answer skips the
    reseed, so the statement runs."""
    seg = SEG.replace("import functools, time\n", "import functools, random, time\n").replace(
        "def seg(n):\n", "def seg(n):\n    random.seed(n)\n"
    )
    cells = [seg, FILL]
    _run(cash_magics, cells)
    reasons = _reasons(monkeypatch)
    seen = _run(cash_magics, cells, [1])
    assert seen == [["COMPUTED"]], (seen, reasons)
    assert "already holds an answer the statement computed" in str(reasons[cells[-1]]), reasons
