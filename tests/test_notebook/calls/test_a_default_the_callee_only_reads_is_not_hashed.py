"""A default the function only reads is not hashed around every miss.

``def f(x, buf=BIG): return x + float(buf[0])`` with a 40 MB array: the
default was hashed before and after each miss, to see whether the call
changed it, 9 ms a call where a global read the same way costs nothing.
Every way of changing it, or of handing it on, is still watched.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.notebook.call_key import callee_watched_defaults

BIG = np.zeros(4)


class Box:
    def __add__(self, other):
        return other


def reads_an_item(x, buf=BIG):
    return x + float(buf[0])


def reads_a_shape(x, buf=BIG):
    return f"{x}{buf.shape}" if len(buf) > x else abs(buf.T[0] * 2)


def adds_to_an_item(x, buf=BIG):
    buf[0] += 1
    return x


def adds_in_place(x, buf=BIG):
    buf += 1
    return x


def calls_a_method(x, buf=BIG):
    buf.fill(x)
    return x


def hands_it_on(x, buf=BIG):
    np.copyto(buf, x)
    return x


def binds_another_name(x, buf=BIG):
    b = buf
    b[0] = x
    return x


def iterates(x, buf=BIG):
    for row in buf:
        row += x
    return x


def a_lambda_holds_it(x, buf=BIG):
    put = lambda: buf.fill(x)
    put()
    return x


def returns_it(x, buf=BIG):
    return buf


def a_users_value(x, box=Box()):
    return box + x


@pytest.mark.parametrize("fn", [reads_an_item, reads_a_shape], ids=lambda f: f.__name__)
def test_a_default_only_read_is_not_watched(fn):
    assert callee_watched_defaults(fn) == ()


@pytest.mark.parametrize(
    "fn",
    [
        adds_to_an_item,
        adds_in_place,
        calls_a_method,
        hands_it_on,
        binds_another_name,
        iterates,
        a_lambda_holds_it,
        returns_it,
        a_users_value,
    ],
    ids=lambda f: f.__name__,
)
def test_a_default_the_call_may_change_is_still_watched(fn):
    assert [param for _function, param, _value in callee_watched_defaults(fn)] == [
        "box" if fn is a_users_value else "buf"
    ]


def test_a_builtin_the_module_shadows_does_not_count_as_reading():
    """``len`` there is a function of the module's own that writes."""
    namespace: dict = {"__name__": __name__}
    exec("def len(v):\n    v[0] = 9\n    return 1\n", namespace)
    fn = type(reads_a_shape)(reads_a_shape.__code__, namespace, "reads_a_shape", reads_a_shape.__defaults__)
    assert [param for _function, param, _value in callee_watched_defaults(fn)] == ["buf"]


def test_a_default_changed_by_a_cell_is_still_read_again(cash_magics):
    """Not watched around the call, the default is still the call's input:
    a cell that writes into it gets the call run again."""
    from tests._cell_driver import run_cash_cell

    run_cash_cell(cash_magics, "import numpy as np, time\nBIG = np.zeros(5)")
    run_cash_cell(cash_magics, "def f(x, buf=BIG):\n    time.sleep(0.05)\n    return x + float(buf[0])")
    run_cash_cell(cash_magics, "a = f(1)")
    run_cash_cell(cash_magics, "BIG[0] = 7")
    run_cash_cell(cash_magics, "b = f(1)")
    run_cash_cell(cash_magics, "c = f(1)")
    ns = cash_magics.shell.user_ns
    assert (ns["a"], ns["b"], ns["c"]) == (1.0, 8.0, 8.0)


@pytest.mark.parametrize("default", ["buf=BIG", "*, buf=BIG", "cfg={'buf': BIG}"])
def test_a_default_the_collector_no_longer_tracks_is_still_read_again(cash_magics, default):
    """A collection stops tracking a tuple or dict that holds only a numpy
    array, and the search for what else holds ``BIG`` never reached ``f``:
    ``BIG[0] = 7`` linked nothing to it, and ``f(1)`` gave 1.0 where plain
    Python gives 8.0."""
    from tests._cell_driver import run_cash_cell

    read = "cfg['buf'][0]" if default.startswith("cfg") else "buf[0]"
    run_cash_cell(cash_magics, "import gc, numpy as np, time\nBIG = np.zeros(5)")
    run_cash_cell(cash_magics, f"def f(x, {default}):\n    time.sleep(0.05)\n    return x + float({read})")
    run_cash_cell(cash_magics, "gc.collect()\na = f(1)")
    run_cash_cell(cash_magics, "BIG[0] = 7")
    run_cash_cell(cash_magics, "b = f(1)")
    ns = cash_magics.shell.user_ns
    assert (ns["a"], ns["b"]) == (1.0, 8.0)
