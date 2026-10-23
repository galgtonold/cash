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
from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

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


def test_a_reader_of_a_draw_from_a_library_counter_a_function_holds_is_not_served_stale(statement_processor, mock_shell):
    _run(statement_processor, "import itertools\nc = itertools.count()\ndef a():\n    return next(c)")
    _run(statement_processor, "v = a()")
    _run(statement_processor, "w = [v]")
    _run(statement_processor, "v = a()")
    _run(statement_processor, "w = [v]")
    assert mock_shell.user_ns["w"] == [1]


@pytest.mark.parametrize(
    "attr",
    [pytest.param("{'x': j}", id="dict attributes"), pytest.param("[1, 2, 3]", id="list attributes")],
)
def test_many_objects_holding_only_numbers_are_served(statement_processor, attr):
    fields = "\n".join(f"        self.p{j} = {attr.replace('j', str(j))}" for j in range(40))
    _run(statement_processor, f"class P:\n    def __init__(self):\n{fields}\nobjs = {{i: P() for i in range(64)}}")
    _run(statement_processor, "def total():\n    return len(objs)")
    _run(statement_processor, "v = total()")
    second = _run(statement_processor, "v = total()")
    assert second["status"] == CacheStatus.RESTORED


_RUN_ALL_DRAWS = {
    "a counter a function reads": [
        "import itertools\nsrc = itertools.count()\ndef g():\n    return next(src)",
        "v = g()",
        "print('v is', v)\nseen = v",
    ],
    "a counter made through __import__": [
        "src = __import__('itertools').count()\ndef g():\n    return next(src)",
        "v = g()",
        "seen = v",
    ],
    "one counter two functions read": [
        "import itertools\nc = itertools.count()\ndef a():\n    return next(c)\ndef b():\n    return next(c) * 10",
        "v = a()",
        "u = b()",
        "seen = [v, u]",
    ],
}


@pytest.mark.parametrize("case", list(_RUN_ALL_DRAWS))
def test_a_reader_below_a_draw_through_a_function_does_not_draw_again(cash_magics, monkeypatch, case):
    """``v = g()`` runs every time, as it draws from the counter ``g``
    reads. The reader below checks that ``v`` is what ``v = g()`` gave; it
    found the simulation's key for ``v = g()`` answering ``src`` from where
    the draw left it, not where it stood, missed the value ``v`` was keyed
    by, and ran ``v = g()`` again: a second draw, ``v`` 1 where Python gives
    0, on every Run All."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = _RUN_ALL_DRAWS[case]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        plain: dict = {}
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
            exec(cell, plain)
        assert ns["seen"] == plain["seen"], f"Run All {run + 1}"


_HELD_IN_AN_OUTPUT = {
    "a pandas Series holding a generator": [
        f"import pandas as pd\ns = pd.Series([0] * 70 + [{_GEN}], dtype=object)",
        "v = next(s.iloc[70])",
        "u = next(s.iloc[70])",
        "seen = [v, u]",
    ],
    "a registry of objects one of which holds a generator": [
        "class O:\n    def __init__(self, g=None):\n        self.g = g\n    def draw(self):\n        return next(self.g)\n"
        f"reg = {{f'o{{j}}': O() for j in range(64)}}\nreg['last'] = O({_GEN})\nk = 'last'\n"
        "def f():\n    return reg[k].draw()",
        "v = f()",
        "seen = [v, 'read']",
    ],
}


@pytest.mark.parametrize("case", list(_HELD_IN_AN_OUTPUT))
def test_a_container_holding_a_generator_is_built_again_on_each_run_all(cash_magics, monkeypatch, case):
    """The statement building the container was served, or skipped as
    current: the container came back holding the generator the readers of
    the earlier runs had drawn from, and the second Run All read 2 and 3
    where Python reads 0 and 1."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = _HELD_IN_AN_OUTPUT[case]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        plain: dict = {}
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
            exec(cell, plain)
        assert ns["seen"] == plain["seen"], f"Run All {run + 1}"


def test_a_statement_binding_a_container_that_holds_a_generator_says_why_it_runs(statement_processor):
    _run(statement_processor, "class O:\n    def __init__(self, g=None):\n        self.g = g")
    _run(statement_processor, "reg = {f'o{j}': O() for j in range(64)}")
    metrics = _run(statement_processor, f"reg['last'] = O({_GEN})")
    assert any("reg['last'].g" in reason for reason in metrics["uncacheable_reasons"])
