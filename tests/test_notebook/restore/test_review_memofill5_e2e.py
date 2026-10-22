"""A served statement over a helper memo gives what a run gives:
a key with no text (an int of 5000 digits) leaves the cell working, a
statement reading ``cache_info`` reruns, keys equal across types (``1``,
``1.0``, ``True``) stay apart, and a helper redefined or a builtin patched
later is followed."""

from __future__ import annotations

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as SLEEP
from tests.test_notebook.restore.test_review_memofill4_e2e import NAMES, _clean, _plain, _restart, _run_all

IMPORTS = "import functools, random, time"


def _cycle2(magics, cells1, cells2, capsys=None):
    """Run All cells1, restart, Run All cells2 (an edited notebook)."""
    first, _ = _run_all(magics, cells1, capsys)
    _restart(magics, NAMES)
    return first, *_run_all(magics, cells2, capsys)


def test_big_int_key(cash_magics, capsys):
    # A key whose repr raises (int over 4300 digits): _calls_of's repr is outside its try.
    cells = [
        IMPORTS,
        "@functools.cache\ndef f(n):\n    return n % 7",
        f"a = (f(10**5000), time.sleep({SLEEP}))[0]",
        "print(a)",
    ]
    _, plain_out = _plain(cells)
    first, _ = _run_all(cash_magics, cells, capsys)
    _restart(cash_magics, NAMES)
    seen, out = _run_all(cash_magics, cells, capsys)
    print("STATUSES", first, seen, "out", out, "plain", plain_out)
    assert _clean(out[-1]) == plain_out[-1]
    assert cash_magics.shell.user_ns.get("a") == 10**5000 % 7


def test_cache_info_read_with_upstream_edit(cash_magics):
    # S reads f.cache_info(); an upstream (fast) cell is edited to fill one more key.
    helper = "@functools.cache\ndef f(n):\n    return n * 2"
    s = f"a = (f(1), f.cache_info().currsize, time.sleep({SLEEP}))[:2]"
    cells1 = [IMPORTS, helper, "x = f(5)", s]
    cells2 = [IMPORTS, helper, "x = f(5) + f(6)", s]
    plain, _ = _plain(cells2)
    first, seen, _ = _cycle2(cash_magics, cells1, cells2)
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, "a", ns.get("a"), "plain", plain["a"])
    assert ns["a"] == plain["a"], f"{seen}"


def test_equal_keys_of_other_types_upstream_edit(cash_magics):
    # 1 vs 1.0 vs True: an upstream fast cell changes which equal key the memo holds.
    helper = "@functools.cache\ndef f(x, y):\n    return repr(x) + str(y)"
    s = f"a = (f(1, 2), time.sleep({SLEEP}))[0]"
    for other in ("1.0", "True"):
        cells1 = [IMPORTS, helper, "p = f(1, 2)", s, "b = f(1, 2)"]
        cells2 = [IMPORTS, helper, f"p = f({other}, 2)", s, "b = f(1, 2)"]
        plain, _ = _plain(cells2)
        first, seen, _ = _cycle2(cash_magics, cells1, cells2)
        ns = cash_magics.shell.user_ns
        print("STATUSES", other, first, seen, "a b", ns.get("a"), ns.get("b"), "plain", plain["a"], plain["b"])
        assert (ns["a"], ns["b"]) == (plain["a"], plain["b"]), f"{other} {seen}"
        _restart(cash_magics, NAMES)
        for n in ("f", "p", "a", "b"):
            cash_magics.shell.user_ns.pop(n, None)


def test_pure_fill_served_then_helper_redefined_to_raise(cash_magics, capsys):
    cells = [
        IMPORTS,
        "@functools.cache\ndef f(n):\n    return n * 2",
        f"a = (f(3), time.sleep({SLEEP}))[0]",
        "@functools.cache\ndef f(n):\n    return 1 // (n - 3)",
        "try:\n    b = f(3)\nexcept ZeroDivisionError:\n    b = 'zde'",
    ]
    plain, _ = _plain(cells)
    first, seen, _ = _cycle2(cash_magics, cells, cells)
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, ns.get("a"), ns.get("b"), plain["a"], plain["b"])
    assert (ns["a"], ns["b"]) == (plain["a"], plain["b"])


def test_builtins_patched_later(cash_magics):
    cells = [
        IMPORTS + ", builtins",
        "@functools.cache\ndef f(s):\n    return oct(s)",
        f"a = (f(10), time.sleep({SLEEP}))[0]",
        "_orig = builtins.oct; builtins.oct = lambda s: 'X'",
        "b = f(10); c = f(11)",
        "builtins.oct = _orig",
    ]
    import builtins

    orig = builtins.oct
    try:
        plain, _ = _plain(cells)
        builtins.oct = orig
        first, seen, _ = _cycle2(cash_magics, cells, cells)
    finally:
        builtins.oct = orig
    ns = cash_magics.shell.user_ns
    print("STATUSES", first, seen, ns.get("b"), ns.get("c"), plain["b"], plain["c"])
    assert (ns["b"], ns["c"]) == (plain["b"], plain["c"])
