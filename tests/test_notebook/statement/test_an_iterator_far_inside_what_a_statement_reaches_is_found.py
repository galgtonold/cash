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
    "list item in a function": (
        f"gens = [{_GEN} for _ in range(65)]\ni = 64\ndef f():\n    return next(gens[i])",
        "f()",
    ),
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
    "dict of numbers then an iterator": (
        f"d = {{f'k{{j}}': j for j in range(64)}}\nd['g'] = {_GEN}\nk = 'g'",
        "next(d[k])",
    ),
    "object in a registry": (
        f"class O:\n    def __init__(self, g=None):\n        self.g = g\n    def draw(self):\n        return next(self.g)\n"
        f"reg = {{f'o{{j}}': O() for j in range(64)}}\nreg['last'] = O({_GEN})\nk = 'last'\ndef f():\n    return reg[k].draw()",
        "f()",
    ),
    "eight calls deep": (
        f"gen = {_GEN}\ndef f0():\n    return next(gen)\n"
        + "".join(f"def f{j}():\n    return f{j - 1}()\n" for j in range(1, 8)),
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
    _run(
        statement_processor,
        f"class N:\n    def __init__(self, inner):\n        self.inner = inner\nn = {_GEN}\nfor _ in range(40):\n    n = N(n)",
    )
    _run(
        statement_processor, "def f():\n    m = n\n    while isinstance(m, N):\n        m = m.inner\n    return next(m)"
    )
    _run(statement_processor, "v = f()")
    second = _run(statement_processor, "v = f()")
    assert second["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["v"] == 1
    assert any("could not finish looking" in r for r in second["uncacheable_reasons"])


def test_a_reader_of_a_draw_from_a_library_counter_a_function_holds_is_not_served_stale(
    statement_processor, mock_shell
):
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


_OWN_NAMES = {
    "a comprehension's index": ("gens = [0, (i for i in range(100))]\nj = 0", "[next(gens[j]) for j in [1]]", [1]),
    "a comprehension's key": (
        "d = {'a': 0, 'b': (i for i in range(100))}\nk = 'a'",
        "[next(d[k]) for k in ['b']]",
        [1],
    ),
    "a generator expression's index": (
        "gens = [0, (i for i in range(100))]\ni = 0",
        "sum(next(gens[i]) for i in [1])",
        1,
    ),
    "a lambda's argument": ("gens = [0, (i for i in range(100))]\ni = 0", "(lambda i: next(gens[i]))(1)", 1),
}


@pytest.mark.parametrize("case", list(_OWN_NAMES))
def test_an_index_the_statement_binds_itself_is_not_the_notebooks(statement_processor, mock_shell, case):
    """``j`` in ``[next(gens[j]) for j in [1]]`` is the comprehension's
    own: the notebook's ``j = 0`` picked ``gens[0]``, a number, and the draw
    from ``gens[1]`` was served."""
    setup, expr, drawn_again = _OWN_NAMES[case]
    _run(statement_processor, setup)
    _run(statement_processor, f"v = {expr}")
    second = _run(statement_processor, f"v = {expr}")
    assert second["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["v"] == drawn_again


_LIBRARY_HOLDERS = {
    "a pandas Series": (f"import pandas as pd\ns = pd.Series([0] * 70 + [{_GEN}], dtype=object)", "next(s.iloc[70])"),
    "a numpy object array": (
        f"import numpy as np\na = np.empty(71, dtype=object)\na[:70] = 0\na[70] = {_GEN}",
        "next(a[70])",
    ),
    "an object holding a Series": (
        f"import pandas as pd\nclass H:\n    def __init__(self):\n        self.s = pd.Series([0] * 70 + [{_GEN}], dtype=object)\n"
        "    def draw(self):\n        return next(self.s.iloc[70])\nh = H()",
        "h.draw()",
    ),
}


@pytest.mark.parametrize("case", list(_LIBRARY_HOLDERS))
def test_a_generator_past_the_first_items_of_a_library_container_is_found(statement_processor, mock_shell, case):
    """Only the first 64 items of a library's container were looked at: the
    user's generator at item 70, read through a function, was served."""
    setup, draw = _LIBRARY_HOLDERS[case]
    _run(statement_processor, f"{setup}\ndef f():\n    return {draw}")
    _run(statement_processor, "v = f()")
    second = _run(statement_processor, "v = f()")
    assert second["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["v"] == 1


_DATA = {
    "dates": "import datetime\nxs = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(2500)]",
    "decimals": "import decimal\nxs = [decimal.Decimal(i) / 3 for i in range(2500)]",
    "numpy scalars": "import numpy as np\nxs = list(np.arange(2500))",
    "dataclass points": "import dataclasses\n@dataclasses.dataclass\nclass P:\n    x: float\n    y: float\nxs = [P(i, 2 * i) for i in range(3000)]",
    "namedtuple rows": "import collections\nRow = collections.namedtuple('Row', 'a b')\nxs = [Row(i, str(i)) for i in range(3000)]",
    "objects of the user's": "class O:\n    def __init__(self, i):\n        self.i = i\nxs = {i: O(i) for i in range(3000)}",
    "nested records": "xs = [{'id': i, 'tags': [{'k': 'a', 'v': [i, 1]}]} for i in range(60_000)]",
    "a set of pairs": "xs = {(i, i + 1) for i in range(300_000)}",
}


@pytest.mark.parametrize("case", list(_DATA))
def test_a_function_over_plain_data_is_served(statement_processor, case):
    """Each date, Decimal, numpy scalar or record cost the look one of its
    2,000 objects, and nested records ran out its items: a function over
    2,500 dates ran every time."""
    _run(statement_processor, f"{_DATA[case]}\ndef f():\n    return len(xs)")
    _run(statement_processor, "v = f()")
    second = _run(statement_processor, "v = f()")
    assert second["status"] == CacheStatus.RESTORED, second["uncacheable_reasons"]


def test_a_method_of_records_that_draws_is_still_found(statement_processor, mock_shell):
    """Records of numbers are not looked into one by one, but their class
    is: its method draws from a generator of the notebook."""
    _run(
        statement_processor,
        f"gen = {_GEN}\nclass O:\n    def __init__(self, i):\n        self.i = i\n    def draw(self):\n        return next(gen)\n"
        "xs = [O(i) for i in range(3000)]\ndef f():\n    return xs[0].draw()",
    )
    _run(statement_processor, "v = f()")
    second = _run(statement_processor, "v = f()")
    assert second["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["v"] == 1
