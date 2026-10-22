"""Review repros for claude/served-memo-fill @5817c84e (end to end, cash_magics).

Place under tests/test_notebook/restore/ and run with pytest -s.
Each wrong-result test asserts what plain Python gives.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, random, time"
SLEEP = ABOVE_PERSISTENCE_FLOOR_S


def _restart(magics, names) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _rows(magics) -> list[dict]:
    return magics.cash_status("dict")["last_cell"]["statements"]


def _statuses(magics) -> list[str]:
    return [str(m.get("status")) for m in _rows(magics)]


def _run_all(magics, cells):
    out = []
    for code in cells:
        run_cash_cell(magics, code)
        out.append(_statuses(magics))
    return out


def _plain(cells):
    ns: dict = {}
    for code in cells:
        exec(code, ns)
    return ns


# ---------------------------------------------------------------- wrong results


def test_answer_list_mutated_in_place_by_later_cell(cash_magics):
    # The read-only run records digest of f(1) == [1]. A later cell appends to
    # the memo's own list. `_item_digests` reuses the stored digests because
    # every key and answer is the very same object -> served x == [1].
    helper = "@functools.cache\ndef f(n):\n    return [n]"
    read = f"x = (list(f(1)), time.sleep({SLEEP}))[0]"
    cells = [IMPORTS, helper, "_ = f(1)", read, read, "f(1).append(99)", read]
    plain = _plain(cells)
    seen = _run_all(cash_magics, cells)
    print("STATUSES", seen)
    assert cash_magics.shell.user_ns["x"] == plain["x"], f"statuses: {seen}"


def test_statement_mutates_the_memo_answer_it_read(cash_magics):
    # A read-only run (hit) whose statement appends to the answer the memo
    # holds: recorded after the run, so the memo "holds what was read" and the
    # entry is served; a run appends again.
    helper = "@functools.cache\ndef f(n):\n    return [n]"
    stmt = f"y = (f(1), f(1).append(5), time.sleep({SLEEP}))[0]\nlen_y = len(y)"
    cells = [IMPORTS, helper, "_ = f(1)", stmt, stmt, stmt]
    plain = _plain(cells)
    seen = _run_all(cash_magics, cells)
    print("STATUSES", seen)
    assert cash_magics.shell.user_ns["len_y"] == plain["len_y"], f"statuses: {seen}"


# ---------------------------------------------------------------- cost


def _cost(cash_magics, monkeypatch, disable):
    from cash.notebook import helper_memos

    if disable:  # cf62039 behaviour: no memo watching at all
        monkeypatch.setattr(helper_memos, "watched", lambda ns: None)
    helper = "@functools.cache\ndef h(n):\n    return n * n"
    loop = f"tot = (sum(h(i) for i in range(50)), time.sleep({SLEEP}))[0]"
    down = f"z = (tot * 2, time.sleep({SLEEP}))[0]"
    cells = [IMPORTS, helper, loop, down]
    names = ["functools", "random", "time", "h", "tot", "z"]
    log = {}
    log["runall1"] = _run_all(cash_magics, cells)
    log["runall2"] = _run_all(cash_magics, cells)
    _restart(cash_magics, names)
    log["restart_runall1"] = _run_all(cash_magics, cells)
    log["restart_runall2"] = _run_all(cash_magics, cells)
    return log


@pytest.mark.parametrize("disable", [False, True], ids=["branch", "no_watch"])
def test_cost_notebook_pattern(cash_magics, monkeypatch, disable):
    log = _cost(cash_magics, monkeypatch, disable)
    for k, v in log.items():
        print("COST", "no_watch" if disable else "branch", k, v)


# ---------------------------------------------------------------- >256 visibility


def test_large_memo_visibility(statement_processor):
    # What the user sees for a statement over a memo of >256 answers: the
    # filling run, the read-only run (stored, never servable), and later runs.
    sp = statement_processor
    for code in [IMPORTS, "@functools.cache\ndef h(n):\n    return n * n"]:
        sp.process_statement(code)
    loop = f"tot = (sum(h(i) for i in range(300)), time.sleep({SLEEP}))[0]"
    for i in range(4):
        m = sp.process_statement(loop)
        print("LARGE", i, {k: m.get(k) for k in ("status", "miss_reason", "uncacheable_reasons", "stored", "warnings")})
    small = f"tot2 = (sum(h(i) for i in range(5)), time.sleep({SLEEP}))[0]"
    m = sp.process_statement(small)
    print("LARGE small-over-big-memo", {k: m.get(k) for k in ("status", "miss_reason", "uncacheable_reasons")})


def test_cache_clear_resets_misses_so_a_filling_run_looks_read_only(cash_magics):
    # cache_clear() resets hits AND misses to 0. A statement that clears and
    # then misses as many times as the memo had missed before leaves both
    # counters equal to the snapshot: `used` sees no change, records nothing,
    # and the entry is stored without helper_memos. Served after a restart,
    # the memo stays empty and the helper body (random.seed) runs below.
    helper = "@functools.cache\ndef cases(n):\n    random.seed(n)\n    return tuple(random.random() for _ in range(n))"
    warm = "_w = cases(7)"
    fill = "def bench_fn():\n    cases.cache_clear()\n    return sum(cases(3))\n"
    slow = f"bench = (bench_fn(), time.sleep({SLEEP}))[0]"
    below = "r = (cases(3), random.random())[1]"
    cells = [IMPORTS, "random.seed(0)", helper, fill, warm, slow, below]
    plain = _plain(cells)
    _run_all(cash_magics, cells)
    _restart(cash_magics, ["functools", "random", "time", "cases", "bench_fn", "bench", "r", "_w"])
    seen = _run_all(cash_magics, cells)
    print("STATUSES", seen)
    assert cash_magics.shell.user_ns["r"] == plain["r"], f"statuses: {seen}"


def test_cache_clear_inside_statement_after_restart_prints(cash_magics, capsys):
    # Same hole, observable effect: the helper prints when its body runs.
    # Plain Python: `below` hits the memo the `slow` run refilled -> prints
    # nothing. cash after a restart serves `slow` (stored without a memo
    # record), the memo keeps only 7, and `below` runs the body.
    helper = "@functools.cache\ndef cases(n):\n    print('BODY', n)\n    return n * 2"
    fill = "def bench_fn():\n    cases.cache_clear()\n    return cases(3)"
    warm = "_w = cases(7)"
    slow = f"bench = (bench_fn(), time.sleep({SLEEP}))[0]"
    below = "r = cases(3)"
    cells = [IMPORTS, helper, fill, warm, slow, below]
    _run_all(cash_magics, cells)
    _restart(cash_magics, ["functools", "random", "time", "cases", "bench_fn", "bench", "r", "_w"])
    seen = []
    for code in cells:
        capsys.readouterr()
        run_cash_cell(cash_magics, code)
        seen.append(_statuses(cash_magics))
    out = capsys.readouterr().out
    print("STATUSES", seen, "below printed:", out.strip().splitlines()[-3:])
    assert "BODY 3" not in out, f"statuses: {seen}; the helper body ran below: {out!r}"


def test_reloaded_module_memo_is_not_watched(cash_magics, capsys, tmp_path, monkeypatch):
    # `_owner_memos` keeps a module's memo list while len(vars(module)) is
    # unchanged. importlib.reload (what %autoreload does) rebinds `f` to a NEW
    # memo in the same dict: the cached list still holds the old one, so a run
    # filling the new memo is not FILLED and is stored without a memo record.
    import sys

    (tmp_path / "memo_rev2_mod.py").write_text(
        "import functools\n@functools.cache\ndef f(n):\n    print('BODY', n)\n    return n * 2\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cells = [
        "import importlib, time\nimport memo_rev2_mod as mm",
        "_a = mm.f(5)",
        "importlib.reload(mm)",
        f"bench = (mm.f(3), time.sleep({SLEEP}))[0]",
        "r = mm.f(3)",
    ]
    try:
        first = _run_all(cash_magics, cells)
        sys.modules.pop("memo_rev2_mod", None)
        _restart(cash_magics, ["importlib", "time", "mm", "_a", "bench", "r"])
        seen = []
        for code in cells:
            capsys.readouterr()
            run_cash_cell(cash_magics, code)
            seen.append(_statuses(cash_magics))
        out = capsys.readouterr().out
    finally:
        sys.modules.pop("memo_rev2_mod", None)
    print("STATUSES", first, seen, "below printed:", out)
    assert "BODY 3" not in out, f"first {first} restart {seen}; helper body ran below: {out!r}"
