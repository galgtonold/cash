"""Review repros for claude/served-memo-fill @e0575c53 (end to end, cash_magics).

Place under tests/test_notebook/restore/ and run with pytest -s -n 0 (or -n 4).
Pattern: Run All, restart, Run All; compare with plain Python. A slow cell
(sleep over the persistence floor) is stored and served after the restart;
a fast one is not persisted and reruns, which is where the memo's state shows.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, random, time"
SLEEP = ABOVE_PERSISTENCE_FLOOR_S
NAMES = ["functools", "random", "time"]


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


def _plain(cells, capsys=None):
    import contextlib
    import io

    ns: dict = {}
    outs = []
    for code in cells:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        outs.append(buf.getvalue())
    return ns, outs


def _clean(text):
    return text.replace("<IPython.core.display.HTML object>\n", "")


def _cycle(magics, cells, capsys=None):
    first, _ = _run_all(magics, cells, capsys)
    _restart(magics, NAMES)
    return first, *_run_all(magics, cells, capsys)


# ---------------------------------------------------------------- (a) purity: default arguments


def test_pure_fill_helper_with_mutable_default(cash_magics):
    # `acc += [k]` mutates the default list without an attribute load, and
    # the answer is an int: `_pure` says pure, the filling runs are stored.
    helper = "@functools.cache\ndef h(k, acc=[]):\n    acc += [k]\n    return len(acc)"
    cells = [
        IMPORTS,
        helper,
        f"a = (h(1), time.sleep({SLEEP}))[0]",
        f"b = (h(2), time.sleep({SLEEP}))[0]",
        "c = h(1) + h(3)",
    ]
    plain, _ = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, "a b c", ns.get("a"), ns.get("b"), ns.get("c"), "plain", plain["c"])
    assert (ns["a"], ns["b"], ns["c"]) == (plain["a"], plain["b"], plain["c"]), f"{seen}"


def test_pure_fill_helper_printing_through_a_default(cash_magics, capsys):
    # `out=print` as a default: the body calls a local, no global is read.
    helper = "@functools.cache\ndef h(k, out=print):\n    out('BODY', k)\n    return k * 2"
    cells = [IMPORTS, helper, f"a = (h(1), time.sleep({SLEEP}))[0]", "b = h(1)"]
    _, plain_out = _plain(cells)
    first, seen, out = _cycle(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "out", out, "plain", plain_out)
    assert _clean(out[-1]) == plain_out[-1], f"{seen}"


def test_pure_fill_helper_whose_callee_is_redefined(cash_magics):
    # h calls a pure user function g; g is redefined after the fill. A run
    # keeps g's old answer in the memo; served, the memo is empty and a later
    # call computes with the new g. `_pure` is asked at lookup, before the
    # redefinition runs, so the fill is served.
    cells = [
        IMPORTS,
        "def g(k):\n    return k * 2",
        "@functools.cache\ndef h(k):\n    return g(k)",
        f"a = (h(1), time.sleep({SLEEP}))[0]",
        "def g(k):\n    return k * 3",
        "b = h(1)",
    ]
    plain, _ = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, "b", ns.get("b"), "plain", plain["b"])
    assert ns["b"] == plain["b"], f"{seen}"


def test_pure_fill_helper_reading_a_shadowed_builtin(cash_magics):
    # Same with a builtin the helper reads, shadowed by a global later.
    cells = [
        IMPORTS,
        "@functools.cache\ndef h(k):\n    return len(k)",
        f"a = (h('abc'), time.sleep({SLEEP}))[0]",
        "def len(s):\n    return 99",
        "b = h('abc')",
    ]
    plain, _ = _plain(cells)
    first, seen, _ = _cycle(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, "b", ns.get("b"), "plain", plain["b"])
    assert ns["b"] == plain["b"], f"{seen}"


# ---------------------------------------------------- (b) a clear the key ids miss (recycled tuple)


def test_unnamed_clear_refill_other_key_reuses_tuple_id(cash_magics, capsys):
    # The clear is not named (`'cache' + chr(95) + 'clear'`), and the refill
    # with key (1, 3) gets the freed key tuple (1, 2)'s address; the answer is
    # the same small int: counters and held ids equal, so the statement is
    # stored as not touching h. Served, h keeps (1, 2), so the last cell hits
    # where a run misses and prints.
    helper = "@functools.cache\ndef h(a, b):\n    print('BODY', a, b)\n    return 0"
    cells = [
        IMPORTS,
        helper,
        f"a = (h(1, 2), time.sleep({SLEEP}))[0]",
        f"b = (getattr(h, 'cache' + chr(95) + 'clear')(), h(1, 3), time.sleep({SLEEP}))[1]",
        "c = h(1, 2)",
    ]
    _, plain_out = _plain(cells)
    first, seen, out = _cycle(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "out", out, "plain", plain_out)
    assert _clean(out[-1]) == plain_out[-1], f"{seen}"


# ---------------------------------------------- (c) a module function made a memo inside the statement


def test_module_function_made_a_memo_and_filled_in_one_statement(cash_magics, capsys, tmp_path, monkeypatch):
    # Fails ~3 in 8 runs at e0575c53, never at cf62039 (8/8 pass). Cause: the
    # call `functools.cache(mf4_mod.g)` is stored by the call unit because
    # call_unit._run_miss times `helper_memos.watched()` (~2 ms the first time
    # in a statement) inside `elapsed`, pushing it over the 3 ms call floor;
    # after the restart that call is served (held memo object, already filled)
    # and BODY is not printed. Also: `_where` re-reads modules once per
    # statement, so the memo bound and filled here is not seen by `used`.
    (tmp_path / "mf4_mod.py").write_text("def g(n):\n    print('BODY', n)\n    return n * 2\nf = g\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    cells = [
        IMPORTS + ", mf4_mod",
        f"a = (setattr(mf4_mod, 'f', functools.cache(mf4_mod.g)), mf4_mod.f(1), time.sleep({SLEEP}))[1]",
        "b = mf4_mod.f(1)",
    ]
    import sys

    try:
        _, plain_out = _plain(cells)
        sys.modules.pop("mf4_mod", None)
        first, _ = _run_all(cash_magics, cells, capsys)
        _restart(cash_magics, NAMES + ["mf4_mod"])
        sys.modules.pop("mf4_mod", None)
        seen, out = _run_all(cash_magics, cells, capsys)
    finally:
        sys.modules.pop("mf4_mod", None)
    print("STATUSES", first, seen, "out", out, "plain", plain_out)
    assert [_clean(o) for o in out] == plain_out, f"{seen}"


def test_module_function_made_a_memo_on_a_slow_machine(cash_magics, capsys, tmp_path, monkeypatch):
    """On a busy machine the call ``functools.cache(mf4_mod.g)`` timed past
    the call floor and was stored. A memo is kept by reference, so after the
    restart the call handed back the one the first run filled, and ``BODY``
    was not printed. A memo is never served."""
    import time

    from cash import _clock

    real = time.perf_counter
    start = real()
    monkeypatch.setattr(_clock, "_REAL", (lambda: start + (real() - start) * 100,))
    test_module_function_made_a_memo_and_filled_in_one_statement(cash_magics, capsys, tmp_path, monkeypatch)
