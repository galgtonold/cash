"""Review round 6 of serving through a helper memo.

- The forward probe passed a pure memo lacking a key the statement read; a
  statement of the cell above patched a builtin the helper reads and the
  lookup refused, so a variable the probe had left to the entry ran on its
  value in memory (``w = 5`` and NOTEBOOK-INPUT-UNBUILT, where 1 is right).
- ``getattr(f, 'cache_' + 'info')()`` in the statement's own text was served
  with the old counters.
- A name only containing ``cache_info`` (``cache_info_rows``) reran the
  statement every time, with no reason.
- A record of an earlier layout (``calls`` as plain strings) raised.
"""

from __future__ import annotations

import builtins
import warnings

from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook import helper_memos
from tests._cell_driver import run_cash_cell
from tests.test_notebook.restore.test_review_memofill4_e2e import _plain
from tests.test_notebook.restore.test_review_memofill5_e2e import IMPORTS, SLEEP, _cycle2

ORIG_OCT = builtins.oct

CELLS = [
    "import functools, builtins\n_o = builtins.oct\n@functools.cache\ndef h(n):\n    return oct(n)",
    "w = 0",
    "builtins.oct = (lambda n: 'X') if getattr(builtins, 'PATCH_OCT', False) else _o\nw = w + len(h(8))",
]


def _codes(magics, cell, cells) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, cell, cells=cells)
    return [c for w in caught if (c := getattr(w.message, "code", None))]


def _st(magics):
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def test_probe_says_hit_lookup_refuses_after_builtin_patch(cash_magics, monkeypatch):
    """Probe (refill=False) passes a pure memo lacking its key; an earlier
    statement of the cell patches oct; the lookup refuses. The cell rerun
    alone should start from w's base (0) as cash does for w -= 1: 0 + len('X') = 1."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    ns = cash_magics.shell.user_ns
    try:
        for c in CELLS:
            run_cash_cell(cash_magics, c, cells=CELLS)
        assert ns["w"] == 4
        # control: rerun alone with nothing changed is served from base
        run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
        assert ns["w"] == 4
        # scratch cells (not part of the notebook): empty the memo, flip mode
        run_cash_cell(cash_magics, "h.cache_clear()")
        builtins.PATCH_OCT = True
        codes = _codes(cash_magics, CELLS[2], CELLS)
        assert ns["w"] == 1, codes
        assert "NOTEBOOK-INPUT-UNBUILT" not in codes
    finally:
        builtins.oct = ORIG_OCT
        builtins.__dict__.pop("PATCH_OCT", None)


def test_cache_info_through_getattr_text(cash_magics):
    helper = "@functools.cache\ndef f(n):\n    return n * 2"
    s = f"a = (f(1), getattr(f, 'cache_' + 'info')().currsize, time.sleep({SLEEP}))[:2]"
    cells1 = [IMPORTS, helper, "x = f(5)", s]
    cells2 = [IMPORTS, helper, "x = f(5) + f(6)", s]
    plain, _ = _plain(cells2)
    first, seen, _ = _cycle2(cash_magics, cells1, cells2)
    ns = cash_magics.shell.user_ns
    assert ns.get("a") == plain["a"] == (2, 3), (first, seen)


def test_cache_info_substring_false_positive(cash_magics):
    cells = [
        IMPORTS,
        "@functools.cache\ndef f(n):\n    return n * 2",
        f"cache_info_rows = (sum(range(10)), time.sleep({SLEEP}))[0]",
    ]
    for c in cells:
        run_cash_cell(cash_magics, c)
    run_cash_cell(cash_magics, cells[2])
    assert _st(cash_magics) == ["RESTORED"]


def test_a_record_of_an_earlier_layout_is_refused_not_raised():
    """``calls`` as ``[text]`` (an earlier commit's layout) is refused."""
    import functools

    @functools.cache
    def f(n):
        return n * 3

    f(10)
    record = {"where": ["ns", "f"], "items": ["gone"], "pure": True, "calls": ["((10,), {})"]}
    f.cache_clear()
    assert helper_memos._refill(f, record["calls"]) is False
