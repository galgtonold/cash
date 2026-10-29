"""A call whose callee changes a global in a way its text does not spell
as a write runs again inside every Run All, as plain Python runs it.

``bisect.insort(LOG, x)``, ``DQ.appendleft(x)``, ``getattr(LOG, 'append')(x)``,
``PUSH(x)`` with ``PUSH = LOG.append``, ``setattr(NS, 'a', x)`` on a class,
and any write of a ``def`` built by ``exec``: the call cache read none of
them as a write, and on the second Run All served the call and left ``LOG``
empty. What the callee hands to code whose effect cannot be read, or loads
when its text cannot be read, is now hashed before and after a miss, as its
arguments are, and a call that changed it is not stored.

The same holds for what a global holds and hands on rather than names:
``PUSHP = functools.partial(list.append, LOG)``, a class's method handed
its object first (``collections.deque.append(DQ, x)``), and a default
(``def inner(x, L=LOG)``), with and without a record of what ran.
"""

from __future__ import annotations

import warnings

import pytest

from cash.notebook import shadow
from tests._cell_driver import run_cash_cell

SETUP = (
    "import operator, bisect, collections, contextvars, functools, queue, threading, time\n"
    "LOG = []\nD = {}\nDQ = collections.deque()\nOD = collections.OrderedDict(a=1, b=2)\n"
    "class NS: pass\n"
    "PUSH = LOG.append\n"
    "CV = contextvars.ContextVar('cv', default=0)\n"
    "Q = queue.Queue()\nSQ = queue.SimpleQueue()\nEV = threading.Event()\n"
    "PUSHP = functools.partial(list.append, LOG)\n"
    "OPS = {'push': LOG.append}\nOPL = [LOG.append]\nHOOKS = [LOG.append]\n"
    "import types\nSNS = types.SimpleNamespace(push=LOG.append)\n"
    "class HC: pass\nHC.push = functools.partial(list.append, LOG)\n"
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
    # Its value lives in the context: a restore of ``CV`` cannot set it.
    "context_variable_set": ("CV.set(CV.get() + x)", "n = CV.get()", 3),
    # A lock inside: they pickle to nothing a change moves.
    "queue_put": ("Q.put(x)", "n = Q.qsize()", 1),
    "simple_queue_put": ("SQ.put(x)", "n = SQ.qsize()", 1),
    "event_set": ("EV.set()", "n = EV.is_set()", True),
    "partial_of_append": ("PUSHP(x)", "n = len(LOG)", 1),
    "class_method_of_a_module": ("collections.deque.append(DQ, x)", "n = len(DQ)", 1),
    "builtin_class_method": ("list.append(LOG, x)", "n = len(LOG)", 1),
    "a_method_a_dict_holds": ("OPS['push'](x)", "n = len(LOG)", 1),
    "a_method_a_list_holds": ("OPL[0](x)", "n = len(LOG)", 1),
    # Called through a local bound to what the global holds.
    "hooks_iterated": ("for h in HOOKS:\n        h(x)", "n = len(LOG)", 1),
    "a_local_alias_of_a_held_method": ("p = OPS['push']\n    p(x)", "n = len(LOG)", 1),
    "an_alias_got_from_a_dict": ("p = OPS.get('push')\n    p(x)", "n = len(LOG)", 1),
    "an_alias_unpacked_from_a_tuple": ("p, q = OPS['push'], None\n    p(x)", "n = len(LOG)", 1),
    "an_alias_from_either_branch": ("p = OPS['push'] if x else None\n    p(x)", "n = len(LOG)", 1),
    "an_alias_bound_by_a_walrus": ("(p := HOOKS[0])(x)", "n = len(LOG)", 1),
    "an_alias_of_a_namespace_attribute": ("p = SNS.push\n    p(x)", "n = len(LOG)", 1),
    "an_alias_got_by_getattr": ("p = getattr(HC, 'push')\n    p(x)", "n = len(LOG)", 1),
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


@pytest.fixture(params=["trace", "no_trace"])
def record(request, monkeypatch):
    if request.param == "no_trace":
        monkeypatch.setenv("CASH_TRACE", "0")
        monkeypatch.setattr(shadow, "SHADOW", None)
    return request.param


HOLDERS = {
    "partial_of_append": "PUSHP(x)",
    "class_method_of_a_module": "collections.deque.append(DQ, x)",
    "builtin_class_method": "list.append(DQ, x)",
    "default_argument": None,
    "default_an_item_of_a_global": None,
    "default_of_a_lambda_it_calls": None,
    "a_partial_a_class_holds": "HP.push(x)",
}
#: The whole definition, where the shape is not one call in ``inner``.
DEFINED = {
    "default_argument": "def inner(x, L=DQ):\n    time.sleep(0.05)\n    L.append(x)\n    return x * 10",
    # Held by ``CFG`` alone: no global is the default itself.
    "default_an_item_of_a_global": (
        "CFG = {'a': collections.deque()}\ndef inner(x, L=CFG['a']):\n    time.sleep(0.05)\n    L.append(x)\n"
        "    return x * 10"
    ),
    "default_of_a_lambda_it_calls": (
        "PL = lambda x, L=DQ: L.append(x)\ndef inner(x):\n    time.sleep(0.05)\n    PL(x)\n    return x * 10"
    ),
}


@pytest.mark.parametrize("shape", list(HOLDERS))
def test_a_change_through_what_a_global_holds_is_made_again(cash_magics, record, shape):
    body = HOLDERS[shape]
    if body is None:
        fn = DEFINED[shape]
    else:
        fn = _readable(body)
    setup = SETUP.replace(
        "DQ = collections.deque()", "DQ = []" if "list." in (body or "") else "DQ = collections.deque()"
    )
    setup += "class HP: pass\nHP.push = functools.partial(collections.deque.append, DQ)\n"
    read = {"partial_of_append": "n = len(LOG)", "default_an_item_of_a_global": "n = len(CFG['a'])"}
    cells = [setup + fn, "v = inner(3)", read.get(shape, "n = len(DQ)")]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["v"], ns["n"]) == (30, 1), f"Run All {run + 1}"


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


#: ``(setup, what reads the change)``: the default changed is one of a
#: function the callee calls.
REACHED_DEFAULTS = {
    "a_helper": ("def g(x, L=[]):\n    L.append(x)\n", "g(x)", "n = len(g.__defaults__[0])"),
    "a_method": (
        "class C:\n    def m(self, x, L=[]):\n        L.append(x)\nc = C()\n",
        "c.m(x)",
        "n = len(C.m.__defaults__[0])",
    ),
    "a_lambda": ("g = lambda x, L=[]: L.append(x)\n", "g(x)", "n = len(g.__defaults__[0])"),
}


@pytest.mark.parametrize("trace", ["on", "off"])
@pytest.mark.parametrize("shape", list(REACHED_DEFAULTS))
def test_a_default_of_a_function_the_callee_calls_is_changed_again(cash_magics, monkeypatch, shape, trace):
    """``f`` calling ``g(x, L=[])``: the call was refused, but the statement
    ``v = f(3)`` around it was stored and served on the next Run All, so
    the fresh ``g``'s default stayed empty."""
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
        monkeypatch.setattr(shadow, "SHADOW", None)
    setup, call, read = REACHED_DEFAULTS[shape]
    cells = [
        f"import time\n{setup}def f(x):\n    time.sleep(0.05)\n    {call}\n    return x * 10",
        "v = f(3)",
        read,
    ]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["v"], ns["n"]) == (30, 1), f"Run All {run + 1}"
