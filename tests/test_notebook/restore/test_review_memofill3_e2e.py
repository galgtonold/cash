"""Review repros for claude/served-memo-fill @ae6a36af (end to end, cash_magics).

Place under tests/test_notebook/restore/ and run with pytest -s -n 0.
Pattern: Run All, restart, Run All; compare with plain Python.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, random, time, os, threading"
SLEEP = ABOVE_PERSISTENCE_FLOOR_S
NAMES = ["functools", "random", "time", "os", "threading"]


def _restart(magics, names) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in list(magics.shell.user_ns):
        if name in names or not name.startswith("_") or name.startswith("_w"):
            if name in names or name not in ("In", "Out", "get_ipython", "exit", "quit"):
                magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> list[str]:
    return [str(m.get("status")) for m in magics.cash_status("dict")["last_cell"]["statements"]]


def _run_all(magics, cells, capsys=None):
    seen, out = [], []
    for code in cells:
        if capsys is not None:
            capsys.readouterr()
        run_cash_cell(magics, code)
        seen.append(_statuses(magics))
        if capsys is not None:
            out.append(capsys.readouterr().out)
    return seen, out


def _plain(cells):
    ns: dict = {}
    for code in cells:
        exec(code, ns)
    return ns


def _cycle(magics, cells, capsys=None):
    first, _ = _run_all(magics, cells)
    _restart(magics, NAMES)
    return first, *_run_all(magics, cells, capsys)


# ------------------------------------------------- (a) pure fills, analyzer calls impure helpers pure


def test_pure_fill_helper_drawing_random(cash_magics):
    # PurityAnalyzer calls `return random.random() + k` clean, the answer is
    # a float: the filling run is STORED. After a restart it is served, the
    # memo stays empty, and `below` (fast, reruns) draws a fresh number.
    helper = "@functools.cache\ndef h(k):\n    return random.random() + k"
    cells = [
        IMPORTS,
        "random.seed(0)",
        helper,
        f"a = (h(1), time.sleep({SLEEP}))[0]",
        "b = h(1)",
        "c = random.random()",
    ]
    plain = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, "a b c", ns.get("a"), ns.get("b"), ns.get("c"))
    assert (ns["a"], ns["b"], ns["c"]) == (plain["a"], plain["b"], plain["c"]), f"{seen}"


def test_pure_fill_helper_reading_rebound_global(cash_magics):
    # `return SCALE * k` is pure, SCALE is rebound between the fill and a
    # later call: a run keeps 3 in the memo, served it computes 5.
    helper = "@functools.cache\ndef h(k):\n    return SCALE * k"
    cells = [IMPORTS, "SCALE = 3", helper, f"a = (h(1), time.sleep({SLEEP}))[0]", "SCALE = 5", "b = h(1)"]
    plain = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    print("STATUSES", first, seen, "b", cash_magics.shell.user_ns.get("b"), "plain", plain["b"])
    assert cash_magics.shell.user_ns["b"] == plain["b"], f"{seen}"


def test_pure_fill_helper_reading_environment(cash_magics):
    helper = "@functools.cache\ndef h(k):\n    return os.environ.get('MF3_X', '') + str(k)"
    cells = [
        IMPORTS,
        "os.environ['MF3_X'] = 'one'",
        helper,
        f"a = (h(1), time.sleep({SLEEP}))[0]",
        "os.environ['MF3_X'] = 'two'",
        "b = h(1)",
    ]
    plain = _plain(cells)
    try:
        first, seen, _ = _cycle(cash_magics, cells)
    finally:
        os_env_pop()
    print("STATUSES", first, seen, "b", cash_magics.shell.user_ns.get("b"), "plain", plain["b"])
    assert cash_magics.shell.user_ns["b"] == plain["b"], f"{seen}"


def os_env_pop():
    import os

    os.environ.pop("MF3_X", None)


def test_pure_fill_observable_through_cache_info(cash_magics):
    # A truly pure helper: the fill itself is observable via cache_info().
    helper = "@functools.cache\ndef h(k):\n    return k * 2"
    cells = [IMPORTS, helper, f"a = (h(1), time.sleep({SLEEP}))[0]", "n = h.cache_info().currsize"]
    plain = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    print("STATUSES", first, seen, "n", cash_magics.shell.user_ns.get("n"), "plain", plain["n"])
    # Documented (known-limitations, "A helper memoised with functools.cache"):
    # a served fill of a pure helper leaves the memo empty, which only its
    # counters show; every answer read from it is the run's.
    assert cash_magics.shell.user_ns["n"] in (plain["n"], 0), f"{seen}"


# ------------------------------------------------- (b) cache_clear not named by the statement

PRINTING = "@functools.cache\ndef cases(n):\n    print('BODY', n)\n    return n * 2"


def _clear_case(cash_magics, capsys, setup, slow):
    # warm fills 7; slow clears and fills 3 -> counters (misses 1, size 1)
    # equal the snapshot, so only `clears` can tell. Plain: below hits.
    cells = [IMPORTS, PRINTING, setup, "_w = cases(7)", f"bench = ({slow}, time.sleep({SLEEP}))[0]", "r = cases(3)"]
    first, seen, out = _cycle(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "below printed:", out[-1])
    assert "BODY 3" not in out[-1], f"first {first} restart {seen}; helper body ran below: {out[-1]!r}"


def test_clear_through_alias_bound_in_another_cell(cash_magics, capsys):
    _clear_case(cash_magics, capsys, "clr = cases.cache_clear", "(clr(), cases(3))[1]")


def test_clear_through_getattr_in_a_user_function(cash_magics, capsys):
    setup = "def bench_fn():\n    getattr(cases, 'cache_' + 'clear')()\n    return cases(3)"
    _clear_case(cash_magics, capsys, setup, "bench_fn()")


def test_clear_through_lambda_from_another_cell(cash_magics, capsys):
    _clear_case(cash_magics, capsys, "kill = lambda: cases.cache_clear()", "(kill(), cases(3))[1]")


def test_clear_in_a_thread_target_alias(cash_magics, capsys):
    setup = "clr = cases.cache_clear\ndef in_thread():\n    t = threading.Thread(target=clr); t.start(); t.join()\n    return cases(3)"
    _clear_case(cash_magics, capsys, setup, "in_thread()")


# ------------------------------------------------- (c) memos the scan does not see


def test_exec_defined_memo(cash_magics, capsys):
    # A memo whose code comes from exec() has co_filename '<string>':
    # `_is_user_memo` is False, so a filling run is stored with no record.
    helper = "exec(\"@functools.cache\\ndef cases(n):\\n    print('BODY', n)\\n    return n * 2\", globals())"
    cells = [IMPORTS, helper, f"bench = (cases(3), time.sleep({SLEEP}))[0]", "r = cases(3)"]
    first, seen, out = _cycle(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "below printed:", out[-1])
    assert "BODY 3" not in out[-1], f"first {first} restart {seen}; helper body ran below: {out[-1]!r}"


def test_memo_rebound_inside_loop_statement(cash_magics, capsys):
    # The statement rebinds the memo name in a loop and fills each new memo.
    helper = PRINTING
    loop = (
        "outs = []\nfor i in range(3):\n"
        "    cases = functools.cache(cases.__wrapped__)\n    outs.append(cases(3))\n"
        f"time.sleep({SLEEP})"
    )
    cells = [IMPORTS, helper, loop, "r = cases(3)"]
    first, seen, out = _cycle(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "below printed:", out[-1])
    assert "BODY 3" not in out[-1], f"first {first} restart {seen}; helper body ran below: {out[-1]!r}"


def test_pure_fill_helper_advancing_a_counter(cash_magics):
    # `next(_ctr)` advances a module-level iterator; the analyzer calls it pure.
    helper = "import itertools\n_ctr = itertools.count()\n@functools.cache\ndef h(k):\n    return next(_ctr) + k"
    cells = [IMPORTS, helper, f"a = (h(1), time.sleep({SLEEP}))[0]", "d = next(_ctr)", "b = h(1)"]
    plain = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    print(
        "STATUSES",
        first,
        seen,
        "a b d",
        ns.get("a"),
        ns.get("b"),
        ns.get("d"),
        "plain",
        plain["a"],
        plain["b"],
        plain["d"],
    )
    assert (ns["a"], ns["b"], ns["d"]) == (plain["a"], plain["b"], plain["d"]), f"{seen}"


def test_pure_fill_helper_reading_a_file(cash_magics, tmp_path):
    p = tmp_path / "mf3.txt"
    p.write_text("one", encoding="utf-8")
    helper = f"@functools.cache\ndef h(k):\n    with open({str(p)!r}) as fh:\n        return fh.read() + str(k)"
    cells = [
        IMPORTS,
        helper,
        f"a = (h(1), time.sleep({SLEEP}))[0]",
        f"with open({str(p)!r}, 'w') as fh:\n    fh.write('two')",
        "b = h(1)",
    ]
    plain = _plain(cells)
    p.write_text("one", encoding="utf-8")
    first, _ = _run_all(cash_magics, cells)
    p.write_text("one", encoding="utf-8")
    _restart(cash_magics, NAMES)
    seen, _ = _run_all(cash_magics, cells)
    print("STATUSES", first, seen, "b", cash_magics.shell.user_ns.get("b"), "plain", plain["b"])
    assert cash_magics.shell.user_ns["b"] == plain["b"], f"{seen}"


# ------------------------------------------------- (a) side effects in a callee of the helper


def test_pure_fill_helper_whose_callee_prints(cash_magics, capsys):
    helper = "def g(k):\n    print('SIDE', k)\n    return k + 1\n@functools.cache\ndef h(k):\n    return g(k) * 2"
    cells = [IMPORTS, helper, f"a = (h(3), time.sleep({SLEEP}))[0]", "r = h(3)"]
    first, seen, out = _cycle(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "below printed:", out[-1])
    assert "SIDE 3" not in out[-1], f"first {first} restart {seen}; callee ran below: {out[-1]!r}"


def test_pure_fill_helper_whose_callee_appends_to_a_global(cash_magics):
    helper = (
        "LOG = []\ndef g(k):\n    LOG.append(k)\n    return k + 1\n@functools.cache\ndef h(k):\n    return g(k) * 2"
    )
    cells = [IMPORTS, helper, f"a = (h(3), time.sleep({SLEEP}))[0]", "r = h(3)", "n = len(LOG)"]
    plain = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    print("STATUSES", first, seen, "n", cash_magics.shell.user_ns.get("n"), "plain", plain["n"])
    assert cash_magics.shell.user_ns["n"] == plain["n"], f"{seen}"


def test_pure_fill_helper_whose_callee_seeds(cash_magics):
    helper = "def g(k):\n    random.seed(k)\n    return k + 1\n@functools.cache\ndef h(k):\n    return g(k) * 2"
    cells = [
        IMPORTS,
        helper,
        f"a = (h(3), time.sleep({SLEEP}))[0]",
        "random.seed(99)",
        "r = (h(3), random.random())[1]",
    ]
    plain = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    print("STATUSES", first, seen, "r", cash_magics.shell.user_ns.get("r"), "plain", plain["r"])
    assert cash_magics.shell.user_ns["r"] == plain["r"], f"{seen}"
