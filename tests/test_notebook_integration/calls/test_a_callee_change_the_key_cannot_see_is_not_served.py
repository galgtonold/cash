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

import pytest

from tests._cell_driver import run_cash_cell

SETUP = (
    "import operator, bisect, collections, contextvars, queue, threading, time\n"
    "LOG = []\nD = {}\nDQ = collections.deque()\nOD = collections.OrderedDict(a=1, b=2)\n"
    "class NS: pass\n"
    "PUSH = LOG.append\n"
    "CV = contextvars.ContextVar('cv', default=0)\n"
    "Q = queue.Queue()\nSQ = queue.SimpleQueue()\nEV = threading.Event()\n"
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
