"""A call reading a routine a module attribute holds is keyed on which
routine that is, so a callee that swaps it is not served the first call's
answer by the second.

``m.get = one``, then ``step`` reads ``m.get`` and sets ``m.get = two``:
plain Python gives ``[1, 2]`` for ``[step(n) for n in ['a', 'a']]``. Both
calls were keyed alike -- the module keeps its lineage within the statement,
and what the key reads of a module's data leaves its functions to the code
walk, which follows their code, not which one the attribute holds -- so the
second was served the first's 1. However the attribute is set: ``setattr``,
``vars(m)[...]``, ``m.__dict__.update``, and on a module with a file.

Nor by code and name alone: the routine swapped in may run the same code on
other data -- a closure or default (``mk(1)``, a ``Random(1)``), the object
a method is bound to (``C(1).get``, ``{'a': 1}.get``, ``'aaa'.count``) --
or be another lambda on the same line, or hold another at the end of a
chain of wrappers of any depth. One that holds what cannot be hashed
cannot be told from another: the call runs every time, and says so.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from cash.exceptions import CashCacheIneffectiveWarning
from tests._cell_driver import run_cash_cell

BASE = "import types\nm = types.ModuleType('m')\n"
FUNCTIONS = BASE + "def one(n):\n    return 1\ndef two(n):\n    return 2\n"

#: How the callee sets the attribute.
SWAPS = {
    "assigned": "{m}.get = two",
    "setattr": "setattr({m}, 'get', two)",
    "vars": "vars({m})['get'] = two",
    "dict update": "{m}.__dict__.update(get=two)",
}

#: What ``one`` and ``two`` are: the same code and name, other data.
PAIRS = {
    "closures": "def mk(v):\n    def get(n):\n        return v\n    return get\none, two = mk(1), mk(2)",
    "closures over a list": (
        "def mk(v):\n    box = [v]\n    def get(n):\n        return box[0]\n    return get\none, two = mk(1), mk(2)"
    ),
    "recursive closures": (
        "def mk(v):\n    def get(n):\n        return v if n else get(1)\n    return get\none, two = mk(1), mk(2)"
    ),
    "deep wrappers": (
        "def mk(v):\n    def get(n):\n        return v\n    for _ in range(12):\n"
        "        get = (lambda f: lambda n: f(n))(get)\n    return get\none, two = mk(1), mk(2)"
    ),
    "defaults": "one = lambda n, v=1: v\ntwo = lambda n, v=2: v",
    "lambdas on one line": "one, two = (lambda n: 1), (lambda n: 2)",
    "random defaults": (
        "import random\none, two = (lambda n, r=random.Random(1): r.random()), (lambda n, r=random.Random(2): r.random())"
    ),
    "bound methods": (
        "class C:\n    def __init__(self, v):\n        self.v = v\n    def get(self, n):\n        return self.v\n"
        "one, two = C(1).get, C(2).get"
    ),
    "dict get": "one, two = {'a': 1}.get, {'a': 2}.get",
    "chainmap get": (
        "import collections\none, two = collections.ChainMap({'a': 1}).get, collections.ChainMap({'a': 2}).get"
    ),
    "list index": "one, two = ['a', 'b'].index, ['b', 'a'].index",
    "str count": "one, two = 'aaa'.count, 'a'.count",
    "partials": "import functools\ndef base(v, n):\n    return v\none, two = functools.partial(base, 1), functools.partial(base, 2)",
    "itemgetters": "import operator\none, two = operator.itemgetter('a'), operator.itemgetter('b')",
    "callable objects": (
        "class K:\n    def __init__(self, v):\n        self.v = v\n    def __call__(self, n):\n        return self.v\n"
        "one, two = K(1), K(2)"
    ),
}

#: An argument each pair's routines answer differently for.
ARGS = {"itemgetters": "{'a': 1, 'b': 2}"}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _cell(module: str = "m", swap: str = SWAPS["assigned"], arg: str = "'a'") -> str:
    return (
        f"{module}.get = one\n"
        "def step(n):\n"
        f"    v = {module}.get(n)\n"
        f"    {swap.format(m=module)}\n"
        "    return v\n"
        f"lv = [step(n) for n in [{arg}, {arg}]]"
    )


def _plain(setup: str, cell: str):
    namespace: dict = {}
    exec(setup, namespace)
    exec(cell, namespace)
    return namespace["lv"]


def _run(cash_magics, setup: str, cell: str) -> None:
    cells = [setup, cell]
    for each in cells:
        run_cash_cell(cash_magics, each, cells=cells)


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("swap", sorted(SWAPS))
def test_the_second_call_reads_the_function_the_first_left(cash_magics, mock_shell, swap):
    mock_shell.user_ns["__name__"] = "__main__"
    _run(cash_magics, FUNCTIONS, _cell(swap=SWAPS[swap]))

    assert mock_shell.user_ns["lv"] == [1, 2]


@pytest.mark.usefixtures("store_everything")
def test_on_a_module_with_a_file_too(cash_magics, mock_shell, tmp_path, monkeypatch):
    mock_shell.user_ns["__name__"] = "__main__"
    (tmp_path / "swapped_get_lib.py").write_text("def get(n):\n    return 0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "swapped_get_lib", raising=False)
    setup = FUNCTIONS + "import swapped_get_lib\n"
    _run(cash_magics, setup, _cell("swapped_get_lib"))

    assert mock_shell.user_ns["lv"] == [1, 2]


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_a_routine_with_the_same_code_on_other_data_is_another(cash_magics, mock_shell, pair):
    mock_shell.user_ns["__name__"] = "__main__"
    setup = BASE + PAIRS[pair]
    cell = _cell(arg=ARGS.get(pair, "'a'"))
    expected = _plain(setup, cell)
    assert expected[0] != expected[1]

    _run(cash_magics, setup, cell)

    assert mock_shell.user_ns["lv"] == expected


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize(
    "cell",
    [
        "m.get = one\nv1 = step('a')\nv2 = step('a')\nlv = [v1, v2]",
        "m.get = one\nv1 = m.get('a')\nswap()\nv2 = m.get('a')\nlv = [v1, v2]",
        "m.get = one\nv1 = read('a')\nswap()\nv2 = read('a')\nlv = [v1, v2]",
    ],
    ids=["the callee in two statements", "a swap between two reads", "a swap between two calls of a reader"],
)
def test_a_swap_between_statements_is_read_too(cash_magics, mock_shell, cell):
    mock_shell.user_ns["__name__"] = "__main__"
    setup = FUNCTIONS + (
        "def step(n):\n    v = m.get(n)\n    m.get = two\n    return v\n"
        "def swap():\n    m.get = two\n"
        "def read(n):\n    return m.get(n)\n"
    )
    _run(cash_magics, setup, cell)

    assert mock_shell.user_ns["lv"] == [1, 2]


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("pair", ["functions", "recursive closures", "deep wrappers", "bound methods", "dict get"])
def test_a_routine_left_in_place_still_serves_the_second_call(cash_magics, mock_shell, tmp_path, pair):
    """The control: nothing swapped, the second call is served."""
    mock_shell.user_ns["__name__"] = "__main__"
    setup = FUNCTIONS if pair == "functions" else BASE + PAIRS[pair]
    log = tmp_path / "runs.log"
    cell = (
        f"m.get = one\nLOG = {str(log)!r}\n"
        "def step(n):\n"
        "    with open(LOG, 'a') as f:  # @cash:assume-safe\n"
        "        f.write('x')\n"
        "    return m.get(n)\n"
        "lv = [step(n) for n in ['a', 'a']]"
    )
    _run(cash_magics, setup, cell)

    assert mock_shell.user_ns["lv"] == _plain(setup, cell.replace(repr(str(log)), repr(str(tmp_path / "plain.log"))))
    assert log.read_text() == "x"


@pytest.mark.usefixtures("store_everything")
def test_a_routine_holding_what_cannot_be_hashed_runs_every_time_and_says_so(cash_magics, mock_shell, tmp_path):
    mock_shell.user_ns["__name__"] = "__main__"
    setup = BASE + (
        "class Opaque:\n    def __reduce__(self):\n        raise TypeError('no')\n"
        "class H:\n    def __init__(self, v):\n        self.v = v\n        self.o = Opaque()\n"
        "    def get(self, n):\n        return self.v\n"
        "one, two = H(1).get, H(2).get\n"
    )
    log = tmp_path / "runs.log"
    cell = (
        f"m.get = one\nLOG = {str(log)!r}\n"
        "def step(n):\n"
        "    with open(LOG, 'a') as f:  # @cash:assume-safe\n"
        "        f.write('x')\n"
        "    return m.get(n)\n"
        "lv = [step(n) for n in ['a', 'a']]"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cash_magics, setup, cell)

    assert mock_shell.user_ns["lv"] == [1, 1]
    assert log.read_text() == "xx"
    assert any(issubclass(w.category, CashCacheIneffectiveWarning) and "m.get" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]
