"""Per-call cost of helper_memos.watched on the call path (one scan of the
function's globals + every user module, per call executed). Place under
tests/test_notebook/calls/ and run with -rP -n 0."""

import time

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize("names", [100, 1000, 3000])
@pytest.mark.parametrize("mode", ["branch", "no_watch"])
def test_loop_of_call_misses(cash_magics, monkeypatch, mode, names):
    from cash.notebook import helper_memos

    calls = []
    real = helper_memos.watched
    spent = [0.0]

    def timed(ns):
        calls.append(1)
        if mode == "no_watch":
            return None
        t = time.perf_counter()
        try:
            return real(ns)
        finally:
            spent[0] += time.perf_counter() - t

    monkeypatch.setattr(helper_memos, "watched", timed)
    cash_magics.shell.user_ns.update({f"pad{i}": i for i in range(names)})
    run_cash_cell(cash_magics, "import time\ndef g(i):\n    time.sleep(0.002)\n    return i * i")
    t = time.perf_counter()
    run_cash_cell(cash_magics, "out = []\nfor i in range(300):\n    out.append(g(i))")
    dt = time.perf_counter() - t
    print(
        f"CALLCOST {mode} names={names}: {dt:.2f}s, watched() calls={len(calls)}, in watched {spent[0] * 1000:.0f} ms"
    )
