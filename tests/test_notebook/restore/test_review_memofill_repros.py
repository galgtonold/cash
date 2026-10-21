"""Review repros for claude/served-memo-fill @92407c3b (end to end, cash_magics).

Each test asserts what plain Python does; a failure is a served entry where
the memo did not hold the run's answers.
Place under tests/test_notebook/restore/ and run with pytest.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, random, time"


def _restart(magics, names) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> list[str]:
    return [str(m.get("status")) for m in magics.cash_status("dict")["last_cell"]["statements"]]


def test_bounded_lru_returning_a_dict_other_key_after_restart(cash_magics):
    # lru_cache(maxsize=N): gc.get_referents yields the linked list's
    # results before the cache dict; `_cache_dict` takes the result dict
    # {'v': ...}, so "keys" are the digest of 'v' for every answer.
    square = "@functools.lru_cache(maxsize=16)\ndef square(n):\n    return {'v': n * n}"
    fill = f"x = (square(3)['v'], time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    for code in [IMPORTS, square, fill]:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "random", "time", "square", "x"])
    for code in [IMPORTS, square, "other = square(4)", fill]:
        run_cash_cell(cash_magics, code)
    assert _statuses(cash_magics) == ["COMPUTED"], "memo holds 4, not 3: must rerun"


def test_cache_clear_then_refill_other_keys_same_session(cash_magics):
    # CONTROL (passes): a top-level `square.cache_clear()` statement moves
    # square's lineage, so the stale stamp is never consulted. The two
    # variants below clear from inside a function and are served.
    # cache_clear() resets misses: after a refill with as many other keys,
    # (currsize, misses) equals the stamp `_digests` holds and the stale
    # digests of the run's keys are reused.
    square = "@functools.cache\ndef square(n):\n    print('BODY', n)\n    return n * n"
    fill = f"x = (square(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    for code in [IMPORTS, square, fill, "square.cache_clear()", "other = square(4)", fill]:
        run_cash_cell(cash_magics, code)
    assert _statuses(cash_magics) == ["COMPUTED"], "memo holds 4, not 3: must rerun"


def test_memo_under_another_decorator_after_restart(cash_magics):
    helper = (
        "def logged(fn):\n    @functools.wraps(fn)\n    def w(*a):\n        return fn(*a)\n    return w\n"
        "@logged\n@functools.cache\ndef cases(n):\n"
        "    random.seed(n)\n    return tuple(random.random() for _ in range(n))"
    )
    slow = f"bench = (sum(cases(3)) + random.random(), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    below = "r = (cases(3), random.random())[1]"
    cells = [IMPORTS, "random.seed(0)", helper, slow, below]
    plain: dict = {}
    for code in cells:
        exec(code, plain)
    for code in cells:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "random", "time", "cases", "logged", "bench", "r"])
    statuses = []
    for code in cells:
        run_cash_cell(cash_magics, code)
        statuses.append(_statuses(cash_magics))
    assert cash_magics.shell.user_ns["r"] == plain["r"], f"statuses: {statuses}"


def test_memo_as_staticmethod_after_restart(cash_magics):
    helper = (
        "class Cases:\n    @staticmethod\n    @functools.cache\n    def make(n):\n"
        "        random.seed(n)\n        return tuple(random.random() for _ in range(n))"
    )
    slow = f"bench = (sum(Cases.make(3)) + random.random(), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    below = "r = (Cases.make(3), random.random())[1]"
    cells = [IMPORTS, "random.seed(0)", helper, slow, below]
    plain: dict = {}
    for code in cells:
        exec(code, plain)
    for code in cells:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "random", "time", "Cases", "bench", "r"])
    statuses = []
    for code in cells:
        run_cash_cell(cash_magics, code)
        statuses.append(_statuses(cash_magics))
    assert cash_magics.shell.user_ns["r"] == plain["r"], f"statuses: {statuses}"


def test_same_keys_different_answers_after_restart(cash_magics):
    # Only key digests are recorded, not the answers: a memo holding the
    # run's key with another answer passes the check.
    cells_first = [
        IMPORTS,
        "G = 1",
        "@functools.cache\ndef f(n):\n    return n + G",
        f"y = (f(1), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
    ]
    cells_second = [
        IMPORTS,
        "G = 1",
        "@functools.cache\ndef f(n):\n    return n + G",
        "G = 5\n_ = f(1)",
        "G = 1",
        f"y = (f(1), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
    ]
    plain: dict = {}
    for code in cells_second:
        exec(code, plain)
    for code in cells_first:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "random", "time", "f", "G", "y", "_"])
    statuses = []
    for code in cells_second:
        run_cash_cell(cash_magics, code)
        statuses.append(_statuses(cash_magics))
    assert cash_magics.shell.user_ns["y"] == plain["y"], f"statuses: {statuses}"


def _variant(cash_magics, cells, monkeypatch):
    from cash.notebook.statement import freshness

    real = freshness.memos_unfilled
    calls = []
    monkeypatch.setattr(freshness, "memos_unfilled", lambda r, ns: calls.append(real(r, ns)) or calls[-1])
    seen = []
    for code in cells:
        run_cash_cell(cash_magics, code)
        seen.append(_statuses(cash_magics))
    print("VARIANT", seen, "unfilled-calls", calls)
    return seen


def test_clear_via_function(cash_magics, monkeypatch):
    square = "@functools.cache\ndef square(n):\n    print('BODY', n)\n    return n * n"
    fill = f"x = (square(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    seen = _variant(
        cash_magics,
        [
            IMPORTS,
            square,
            "def reset():\n    square.cache_clear()\n    return square(4)",
            fill,
            "other = reset()",
            fill,
        ],
        monkeypatch,
    )
    assert seen[-1] == ["COMPUTED"]


def test_clear_in_module(cash_magics, monkeypatch, tmp_path):
    (tmp_path / "memo_rev_mod.py").write_text(
        "import functools\n@functools.cache\ndef square(n):\n    print('BODY', n)\n    return n * n\n"
        "def reset():\n    square.cache_clear()\n    return square(4)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    fill = f"x = (square(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    try:
        seen = _variant(
            cash_magics, [IMPORTS, "from memo_rev_mod import square, reset", fill, "other = reset()", fill], monkeypatch
        )
    finally:
        import sys

        sys.modules.pop("memo_rev_mod", None)
    assert seen[-1] == ["COMPUTED"]
