"""A call whose callee changes a global in a way its text does not spell
as a write runs again inside every Run All, as plain Python runs it.

``bisect.insort(LOG, x)``, ``DQ.appendleft(x)``, ``getattr(LOG, 'append')(x)``,
``PUSH(x)`` with ``PUSH = LOG.append``, ``setattr(NS, 'a', x)`` on a class,
and any write of a ``def`` built by ``exec``: the call cache read none of
them as a write, and on the second Run All served the call and left ``LOG``
empty. What the callee hands to code whose effect cannot be read, or loads
when its text cannot be read, is now hashed before and after a miss, as its
arguments are, and a call that changed it is not stored.
"""

from __future__ import annotations

import warnings

import pytest

from cash.notebook import shadow
from tests._cell_driver import run_cash_cell

SETUP = (
    "import operator, bisect, collections, time\n"
    "LOG = []\nD = {}\nDQ = collections.deque()\nOD = collections.OrderedDict(a=1, b=2)\n"
    "class NS: pass\n"
    "PUSH = LOG.append\n"
)

BODIES = {
    "getattr_append": ("getattr(LOG, 'append')(x)", "n = len(LOG)", 1),
    "operator_setitem": ("operator.setitem(D, x, x)", "n = len(D)", 1),
    "bound_method_global": ("PUSH(x)", "n = len(LOG)", 1),
    "deque_appendleft": ("DQ.appendleft(x)", "n = len(DQ)", 1),
    "bisect_insort": ("bisect.insort(LOG, x)", "n = len(LOG)", 1),
    "setattr_on_a_class": ("setattr(NS, 'a', x)", "n = getattr(NS, 'a', None)", 3),
    "od_move_to_end": ("OD.move_to_end('a')", "n = list(OD)", ["b", "a"]),
    "operator_iadd": ("operator.iadd(LOG, [x])", "n = len(LOG)", 1),
    "methodcaller": ("operator.methodcaller('append', x)(LOG)", "n = len(LOG)", 1),
}


def _readable(body: str) -> str:
    return f"def inner(x):\n    time.sleep(0.05)\n    {body}\n    return x * 10"


def _built_by_exec(body: str) -> str:
    return f"exec({_readable(body)!r})"


@pytest.mark.parametrize("spelled", [_readable, _built_by_exec], ids=["readable", "exec"])
@pytest.mark.parametrize("shape", list(BODIES))
def test_each_run_all_makes_the_change_again(cash_magics, shape, spelled):
    body, read, expected = BODIES[shape]
    cells = [SETUP + spelled(body), "v = inner(3)", read]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["v"], ns["n"]) == (30, expected), f"Run All {run + 1}"


# Off in a kernel of its own: a warm one was started with traces on, and
# ``CASH_TRACE`` set now never reaches it.
@pytest.mark.parametrize("trace", ["on", pytest.param("off", marks=pytest.mark.fresh_kernel)])
@pytest.mark.parametrize(
    "define",
    [
        "exec('def inner(x, L=LOG):\\n    time.sleep(0.05)\\n    L.append(x)\\n    return x * 10')",
        "inner = eval('lambda x, L=LOG: (time.sleep(0.05), L.append(x), x * 10)[2]')",
    ],
    ids=["exec_def", "lambda_from_a_string"],
)
def test_a_default_an_unreadable_callee_changes_is_made_again(nb_runner, monkeypatch, define, trace):
    """``exec('def inner(x, L=LOG): L.append(x)')``: only the globals it
    loads were watched, and ``L`` is not one, so the call was served and
    ``LOG`` left empty while traces were off."""
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
    nb_runner.create_notebook(["import time\nLOG = []\n" + define, "v = inner(3)", "print('N', len(LOG))"])
    nb_runner.start_kernel()
    for run in range(3):
        nb_runner.run_all()
        assert "N 1" in nb_runner.get_output(3), f"Run All {run + 1}"


#: ``(setup, the default, what reads the change)``: a default no global
#: holds within a lookup's reach.
UNHELD_DEFAULTS = {
    "its_own": ("", "[]", "n = len(inner.__defaults__[0])"),
    "a_class_attribute": ("class K:\n    pass\nK.l = []\n", "K.l", "n = len(K.l)"),
    "three_levels_down": ("CFG = {'a': {'b': {'c': []}}}\n", "CFG['a']['b']['c']", "n = len(CFG['a']['b']['c'])"),
    "a_deque_item": ("H = collections.deque([[]])\n", "H[0]", "n = len(H[0])"),
    "a_tuple_subclass_item": ("class T(tuple):\n    pass\nS = T(([],))\n", "S[0]", "n = len(S[0])"),
}


@pytest.mark.parametrize("spelled", ["readable", "exec"])
@pytest.mark.parametrize("shape", list(UNHELD_DEFAULTS))
def test_a_default_no_global_holds_is_watched_itself(cash_magics, monkeypatch, shape, spelled):
    """``def inner(x, L=K.l): L.append(x)`` with traces off: the defaults
    watched were those a global holds two levels down at most, so the call
    was served and ``K.l`` left empty."""
    monkeypatch.setenv("CASH_TRACE", "0")
    monkeypatch.setattr(shadow, "SHADOW", None)
    setup, default, read = UNHELD_DEFAULTS[shape]
    fn = f"def inner(x, L={default}):\n    time.sleep(0.05)\n    L.append(x)\n    return x * 10"
    cells = [
        "import time, collections\n" + setup + (fn if spelled == "readable" else f"exec({fn!r})"),
        "v = inner(3)",
        read,
    ]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["v"], ns["n"]) == (30, 1), f"Run All {run + 1}"


def test_a_memo_default_is_named_once(cash_magics):
    """``def fib(n, memo={})`` fills its memo on every miss: the call runs
    every time, and says why once, naming the default."""
    cells = [
        "import time\ndef fib(n, memo={}):\n    time.sleep(0.05)\n    memo[n] = n\n    return n",
        "a = fib(3) + fib(4)",
    ]
    said = []
    for _ in range(2):
        for cell in cells:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_cash_cell(cash_magics, cell, cells=cells)
            said += [str(w.message) for w in caught if getattr(w.message, "code", None) == "STORE-DEFAULT-CHANGED"]
    assert cash_magics.shell.user_ns["a"] == 7
    assert len(said) == 1, said
    assert "fib changes its default 'memo'" in said[0]
