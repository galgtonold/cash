"""A function of the notebook's that a statement reaches through a name
spelled at run time -- ``globals()['make']``, ``eval('make')``,
``vars()[name]`` -- is not named in the code, so what it reads was left out
of the key: change ``cfg`` and rerun, and cash served the result computed
with the old ``cfg``.
"""

from __future__ import annotations

import os
import sys
import types
import warnings
from unittest.mock import patch

import pytest

from cash.notebook.cache_key import unkeyed_reach
from cash.notebook.cache_status import CacheStatus
from cash.notebook.string_lookups import code_looks_up_by_string, statement_looks_up_by_string

DEFINITIONS = """import time
cfg = 1
name = 'make'
def make(x):
    time.sleep(0.05)
    return x + cfg
"""

CASES = {
    "globals-in-a-helper": ("def g():\n    return globals()['make'](3)", "r = g()"),
    "globals-get-in-a-helper": ("def g():\n    return globals().get(name)(3)", "r = g()"),
    "globals-in-the-statement": ("pass", "r = globals()['make'](3)"),
    "eval-in-the-statement": ("pass", "r = eval('make')(3)"),
    "eval-in-a-helper": ("def g():\n    return eval('make(3)')", "r = g()"),
    "exec-in-a-helper": (
        "def g():\n    out = {}\n    exec('out[0] = make(3)', globals(), {'out': out})\n    return out[0]",
        "r = g()",
    ),
    "vars-in-the-statement": ("pass", "r = vars()[name](3)"),
    "locals-in-the-statement": ("pass", "r = locals()['make'](3)"),
}


def _run(magics, cells):
    """Run *cells* in order as one Run All; the statuses of the last one's statements."""
    statuses = []
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
    ):
        for cell in cells:
            result = magics._cell_executor.execute_cell(cell, ttl=None, cell_id=magics.resolve_cell_id())
            magics._finalize_cell_body(cell, result)
            statuses = [CacheStatus.parse(m.get("status")) for m in result.all_metrics if m.get("code") == cells[-1]]
    return statuses


@pytest.fixture
def run(cash_magics):
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield lambda cells: _run(cash_magics, cells)


@pytest.mark.parametrize("case", list(CASES))
def test_a_changed_global_of_the_function_is_seen(run, mock_shell, case):
    helper, statement = CASES[case]
    run([DEFINITIONS, helper, statement])
    assert mock_shell.user_ns["r"] == 4
    run([DEFINITIONS.replace("cfg = 1", "cfg = 2"), helper, statement])
    assert mock_shell.user_ns["r"] == 5


def test_a_function_called_by_its_name_is_still_served(run, mock_shell):
    """Control: ``make(3)`` names the function, and its rerun is a hit."""
    cells = [DEFINITIONS, "r = make(3)"]
    assert run(cells) == [CacheStatus.COMPUTED]
    assert run(cells) == [CacheStatus.RESTORED]
    assert mock_shell.user_ns["r"] == 4


LIB = (
    "import time\ncfg = 1\ndef make(x):\n    time.sleep(0.05)\n    return x + cfg\n"
    "def via(name):\n    return globals()[name](3)\n"
)
MODULE_CASES = {
    "getattr-on-the-module": ("pass", "r = getattr(lib, name)(3)"),
    "getattr-in-a-helper": ("def g():\n    return getattr(lib, name)(3)", "r = g()"),
    "vars-of-the-module": ("def g():\n    return vars(lib)[name](3)", "r = g()"),
    "dict-of-the-module": ("def g():\n    return lib.__dict__[name](3)", "r = g()"),
    "the-module-looks-it-up": ("pass", "r = lib.via(name)"),
    "a-module-function-imported-by-name": ("from {lib} import via", "r = via(name)"),
}


@pytest.fixture
def lib_module(tmp_path, monkeypatch):
    name = f"_by_string_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize("case", list(MODULE_CASES))
def test_a_changed_global_of_a_module_function_is_seen(run, mock_shell, lib_module, case):
    """``getattr(lib, name)`` on a module of the user's: the function it
    finds reads the module's ``cfg``, which a cell sets."""
    helper, statement = MODULE_CASES[case]
    helper = helper.replace("{lib}", lib_module)
    setup = f"import {lib_module} as lib\nname = 'make'\nlib.cfg = 1"
    run([setup, helper, statement])
    assert mock_shell.user_ns["r"] == 4
    run([setup.replace("lib.cfg = 1", "lib.cfg = 2"), helper, statement])
    assert mock_shell.user_ns["r"] == 5


def test_it_runs_every_time_and_says_why(cash_magics, cash_instance, mock_shell):
    """No entry is stored for it, the call inside it included: none could
    ever be found again."""
    cash_magics.cash_on("")
    cells = [DEFINITIONS, CASES["globals-in-a-helper"][0], "r = g()"]
    reasons = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (
            patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
            patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        ):
            for _ in range(2):
                for cell in cells:
                    result = cash_magics._cell_executor.execute_cell(
                        cell, ttl=None, cell_id=cash_magics.resolve_cell_id()
                    )
                    cash_magics._finalize_cell_body(cell, result)
                    reasons += [m for m in result.all_metrics if m.get("code") == "r = g()"]
    assert [CacheStatus.parse(m["status"]) for m in reasons] == [CacheStatus.COMPUTED] * 2
    assert "'g' looks names up by a string" in reasons[-1]["uncacheable_reasons"][0]
    assert not [k for k in cash_instance.backend._store if k.startswith("call:")]


def _namespace(source: str) -> dict:
    """*source* run as the notebook's cells are: in a module's namespace,
    which ``me`` names (as ``import __main__`` would)."""
    module = types.ModuleType("nb_by_string")
    module.__dict__["me"] = module
    exec(compile(source, "<nb>", "exec"), module.__dict__)
    return module.__dict__


@pytest.mark.parametrize(
    "body",
    [
        "return globals()['make'](3)",
        "return eval('make(3)')",
        "return NS['make'](3)",
        "return getattr(me, name)(3)",
        "return vars(me)[name](3)",
        "return sys.modules[__name__].make(3)",
        "return inspect.currentframe().f_back.f_globals['make'](3)",
        "return [eval(n) for n in ('make',)][0](3)",
    ],
)
def test_a_lookup_by_a_string_is_seen(body):
    ns = _namespace(
        f"import sys, inspect\nname = 'make'\nNS = globals()\ndef make(x):\n    return x\ndef g():\n    {body}\n"
    )
    assert code_looks_up_by_string(ns["g"].__code__, ns)


@pytest.mark.parametrize(
    "body",
    [
        "return make(3)",
        "return getattr(obj, name)",
        "return vars(obj)",
        "return obj.__dict__[name]",
        "return [m for m in obj.modules()]",
        "return locals()",
        "return vars()",
        "return getattr(sys, 'version')",
    ],
)
def test_what_reads_no_namespace_by_a_string_is_not_seen(body):
    """Controls: an object's attributes, a function's own locals, a
    library model's ``modules()``."""
    ns = _namespace(f"import sys\nname = 'make'\nobj = object()\ndef make(x):\n    return x\ndef g():\n    {body}\n")
    assert not code_looks_up_by_string(ns["g"].__code__, ns)


@pytest.mark.parametrize(
    ("statement", "looks_up"),
    [
        ("r = eval('make')(3)", True),
        ("r = globals()['make'](3)", True),
        ("r = vars()[name](3)", True),
        ("r = locals()['make'](3)", True),
        ("r = [eval(n)(3) for n in ('make',)]", True),
        ("exec('r = globals()[name](3)')", True),
        ("r = make(3)", False),
        ("r = vars(obj)", False),
        # Read as the assignment it runs, its names the statement's own.
        ("exec('w = base * 2')", False),
        # A def runs no body.
        ("def h():\n    return globals()[name]", False),
    ],
)
def test_what_a_statement_looks_up_by_a_string(statement, looks_up):
    ns = _namespace("name = 'make'\nobj = object()\nbase = 1\ndef make(x):\n    return x\n")
    assert statement_looks_up_by_string(statement, ns) is looks_up


def test_a_library_function_looking_names_up_is_not_the_notebooks():
    """Control: ``namedtuple`` reads its caller's ``f_globals``, and
    ``getmodule`` reads ``sys.modules``; neither looks up a name of the
    notebook's, and a statement calling them is keyed as before."""
    ns = _namespace("from collections import namedtuple\nfrom inspect import getmodule\n")
    for statement, inputs in (("P = namedtuple('P', 'a b')", {"namedtuple"}), ("m = getmodule(len)", {"getmodule"})):
        assert unkeyed_reach(statement, inputs, ns, {}) == (set(), set())
