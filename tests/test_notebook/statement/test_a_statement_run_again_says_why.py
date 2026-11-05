"""A statement that runs again, instead of being served, says why.

Each case below once ran again with no reason on its badge ("computed, no
reason given"): a statement too quick to store, one marked
``# @cash:no-cache``, an import, a statement that binds nothing and whose
input moved, one whose entry is gone. The first run of a statement still
carries no reason (`test_miss_reason.py`).
"""

from __future__ import annotations

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests._miss_reason_audit import audit


def _metrics(magics, code: str) -> list[dict]:
    """Run one cell; the metrics of its statements."""
    captured: list[list[dict]] = []
    real = magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real(metrics, **kw)

    magics.badges.render = capture  # type: ignore[assignment]
    try:
        run_cash_cell(magics, code)
    finally:
        magics.badges.render = real  # type: ignore[assignment]
    assert captured, "no metrics captured"
    return captured[-1]


def _last(magics, code: str) -> dict:
    return _metrics(magics, code)[-1]


def test_a_statement_too_quick_to_store_says_so(cash_magics):
    run_cash_cell(cash_magics, "a = 1")
    first = _last(cash_magics, "b = a + 1")
    assert first["status"] == CacheStatus.COMPUTED
    assert first.get("miss_reason") is None
    again = _last(cash_magics, "b = a + 1")
    assert again["status"] == CacheStatus.COMPUTED
    reason = again.get("miss_reason") or ""
    assert reason.startswith("not stored when it last ran under this key: it took "), reason
    assert "a statement must take to be stored" in reason


def test_a_no_cache_statement_says_it_is_marked(cash_magics):
    run_cash_cell(cash_magics, "a = 1")
    m = _last(cash_magics, "# @cash:no-cache\nb = a + 1")
    assert m.get("miss_reason") == "marked # @cash:no-cache, so it runs every time"


def test_an_import_says_it_is_never_stored(cash_magics):
    run_cash_cell(cash_magics, "import json")
    run_cash_cell(cash_magics, "del json")
    m = _last(cash_magics, "import json")
    assert m["status"] == CacheStatus.COMPUTED
    assert (m.get("miss_reason") or "").startswith("an import is never stored"), m.get("miss_reason")


def test_a_statement_binding_nothing_names_the_input_that_moved(cash_magics):
    """``print(a)`` binds nothing, so no record by output name holds what
    it read: its own last run does."""
    run_cash_cell(cash_magics, "a = 1")
    run_cash_cell(cash_magics, "print(a)")
    run_cash_cell(cash_magics, "a = 2")
    m = _last(cash_magics, "print(a)")
    assert m["status"] == CacheStatus.COMPUTED
    assert m.get("miss_reason") == "input changed since it last ran: a", m.get("miss_reason")


def test_a_statement_whose_entry_is_gone_says_so(cash_magics, cash_instance):
    run_cash_cell(cash_magics, "a = 1")
    first = _last(cash_magics, "# @cash:persist\nb = a + 1")
    assert first["storage"], first
    cash_instance.backend.delete(first["cache_key"])
    m = _last(cash_magics, "# @cash:persist\nb = a + 1")
    assert m["status"] == CacheStatus.COMPUTED
    assert (m.get("miss_reason") or "").startswith(
        "the entry stored under this key when it last ran is no longer in the cache"
    ), m.get("miss_reason")


def test_a_changed_function_is_named(cash_magics):
    """``show()`` reads no variable: the function's new text is what moved."""
    run_cash_cell(cash_magics, "def f():\n    return 1")
    run_cash_cell(cash_magics, "def show():\n    return f()")
    run_cash_cell(cash_magics, "show()")
    run_cash_cell(cash_magics, "def f():\n    return 2")
    m = _last(cash_magics, "show()")
    assert m["status"] == CacheStatus.COMPUTED
    assert m.get("miss_reason"), m


def test_cells_run_twice_leave_no_repeat_without_a_reason(cash_magics):
    """The invariant over a small notebook run twice: every statement that
    runs again says why."""
    cells = [
        "import math\nvalues = [1, 2, 3]",
        "total = sum(values)\nprint(total)",
        "# @cash:no-cache\nstamp = len(values)",
        "for v in values:\n    total += math.sqrt(v)",
        "values.append(4)",
        "squares = [v * v for v in values]",
    ]
    seen: set[str] = set()
    found = []
    for _ in range(2):
        for cell in cells:
            found += audit(_metrics(cash_magics, cell), seen)
    assert not found, found


def test_a_pass_over_a_new_loop_value_says_so(cash_magics):
    """``re.finditer`` hands each run new match objects: every pass is one
    the loop has not run before."""
    run_cash_cell(cash_magics, "import re\ntext = 'ab ' * 20")
    loop = "n = 0\nfor m in re.finditer('ab', text):\n    n += 1"
    seen: set[str] = set()
    audit(_metrics(cash_magics, loop), seen)
    again = _metrics(cash_magics, loop)
    assert not audit(again, seen)
    reasons = [m.get("miss_reason") or "" for m in again if m.get("loop_vars")]
    assert any(r.startswith("a new pass: loop variable m holds a value") for r in reasons), reasons
