"""An iterator past the first items or names of what a statement reaches is
found, so the statement runs every time.

The look for an iterator a statement can draw from goes through every item of
the user's own lists, dicts and objects and every name a function reads; it
never samples. Where it cannot finish, the statement runs every time and the
reason says so.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)
_GEN = "(i for i in range(100))"
_ATTRS = "\n".join(f"        self.a{j} = {j}" for j in range(70))
_NAMES = "\n".join(f"    _ = a{j}" for j in range(70))

_CASES = {
    "list item a variable picks": (f"gens = [{_GEN} for _ in range(65)]\ni = 64", "next(gens[i])"),
    "item of a long list": (f"gens = [{_GEN} for _ in range(1000)]\ni = 500", "next(gens[i])"),
    "list item in a function": (f"gens = [{_GEN} for _ in range(65)]\ni = 64\ndef f():\n    return next(gens[i])", "f()"),
    "dict value a variable picks": (f"d = {{f'k{{j}}': {_GEN} for j in range(65)}}\nk = 'k64'", "next(d[k])"),
    "deque item": (f"import collections\ndq = collections.deque({_GEN} for _ in range(65))\ni = 64", "next(dq[i])"),
    "attribute of a list item": (
        f"class O:\n    def __init__(self):\n        self.g = {_GEN}\nobjs = [O() for _ in range(65)]\ni = 64",
        "next(objs[i].g)",
    ),
    "attribute after seventy others": (
        f"class O:\n    def __init__(self):\n{_ATTRS}\n        self.g = {_GEN}\no = O()\ndef f():\n    return next(o.g)",
        "f()",
    ),
    "global after seventy others": (
        "\n".join(f"a{j} = {j}" for j in range(70)) + f"\ngen = {_GEN}\ndef f():\n{_NAMES}\n    return next(gen)",
        "f()",
    ),
    "list of numbers then an iterator": (f"items = [0] * 64 + [{_GEN}]\ni = 64", "next(items[i])"),
    "dict of numbers then an iterator": (f"d = {{f'k{{j}}': j for j in range(64)}}\nd['g'] = {_GEN}\nk = 'g'", "next(d[k])"),
    "object in a registry": (
        f"class O:\n    def __init__(self, g=None):\n        self.g = g\n    def draw(self):\n        return next(self.g)\n"
        f"reg = {{f'o{{j}}': O() for j in range(64)}}\nreg['last'] = O({_GEN})\nk = 'last'\ndef f():\n    return reg[k].draw()",
        "f()",
    ),
    "eight calls deep": (
        f"gen = {_GEN}\ndef f0():\n    return next(gen)\n" + "".join(f"def f{j}():\n    return f{j - 1}()\n" for j in range(1, 8)),
        "f7()",
    ),
}


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


@pytest.mark.parametrize("case", list(_CASES))
def test_a_draw_runs_again_and_advances_the_iterator(statement_processor, mock_shell, case):
    setup, expr = _CASES[case]
    _run(statement_processor, setup)
    _run(statement_processor, f"v = {expr}")
    second = _run(statement_processor, f"v = {expr}")
    assert second["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["v"] == 1


def test_a_list_of_numbers_a_variable_picks_from_is_served(statement_processor):
    _run(statement_processor, "items = list(range(1000))\ni = 500")
    _run(statement_processor, "v = items[i] + 1")
    second = _run(statement_processor, "v = items[i] + 1")
    assert second["status"] == CacheStatus.RESTORED


def test_a_look_that_cannot_finish_runs_every_time_and_says_so(statement_processor, mock_shell):
    _run(statement_processor, f"class N:\n    def __init__(self, inner):\n        self.inner = inner\nn = {_GEN}\nfor _ in range(40):\n    n = N(n)")
    _run(statement_processor, "def f():\n    m = n\n    while isinstance(m, N):\n        m = m.inner\n    return next(m)")
    _run(statement_processor, "v = f()")
    second = _run(statement_processor, "v = f()")
    assert second["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["v"] == 1
    assert any("could not finish looking" in r for r in second["uncacheable_reasons"])
