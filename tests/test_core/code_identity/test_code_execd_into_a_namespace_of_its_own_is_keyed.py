"""Code ``exec``'d into a namespace no module holds is keyed by what it is now.

``ns = {}; exec(rules, ns)``, or a ``types.ModuleType`` never put in
``sys.modules``: the helper walk finds the user's code through a module it
can look up again by name, and code here has none, so the walk took it for
a library's and keyed its name alone. A method replaced on a class defined
there, a helper defined again there, or a class attribute a helper there
reads served the result of the old one; once a call had hit, the fast hit
path served it too, having watched nothing of that code. Plain Python, and
a restart with Run All, give the new result.

A notebook statement that calls such code, uncached, already read it anew:
kept here as the guard that it still does.
"""

from __future__ import annotations

import types

import pytest

from cash import Cash
from tests._cell_driver import run_cash_cell

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RULES = """
class Config:
    factor = 2
    table = [2]

    def m(self, x):
        return x * 2

    @classmethod
    def by_cls(cls, x):
        return x * cls.factor


def helper(x):
    return x * 2


def by_attr(x):
    return x * Config.factor


def by_table(x):
    return x * Config.table[0]


def by_method(x):
    return Config().m(x)


def by_classmethod(x):
    return Config.by_cls(x)


def by_helper(x):
    return helper(x)


def attr_helper(x):
    return x * Config.factor


def by_attr_helper(x):
    return attr_helper(x)


def table_helper(x):
    return x * Config.table[0]


def by_table_helper(x):
    return table_helper(x)


def method_helper(x):
    return Config().m(x)


def by_method_helper(x):
    return method_helper(x)
"""


def _rebind(ns):
    ns["Config"].factor = 5


def _in_place(ns):
    ns["Config"].table[0] = 5


def _replace_method(ns):
    ns["Config"].m = lambda self, x: x * 5


def _redefine_helper(ns):
    exec("def helper(x):\n    return x * 5\n", ns)


CHANGES = [
    ("by_attr", _rebind),
    ("by_table", _in_place),
    ("by_method", _replace_method),
    ("by_method_helper", _replace_method),
    ("by_attr_helper", _rebind),
    ("by_table_helper", _in_place),
    ("by_helper", _redefine_helper),
]


def _namespace(kind: str) -> dict:
    if kind == "named":
        return {"__name__": "cash_ut_rules"}
    if kind == "bare":
        return {}
    return types.ModuleType("cash_ut_rules_module").__dict__  # never registered


@pytest.mark.parametrize("kind", ["named", "bare", "module"])
@pytest.mark.parametrize(("name", "change"), CHANGES, ids=[f"{n}-{c.__name__.strip('_')}" for n, c in CHANGES])
def test_a_change_to_execd_code_or_what_it_reads_is_a_new_result(kind, name, change):
    c = Cash(backend="memory", register_magic=False)
    ns = _namespace(kind)
    exec(RULES, ns)
    f = c.cache(ns[name])
    assert [f(3), f(3), f(3)] == [6, 6, 6]  # hits, the fast path among them
    change(ns)
    assert f(3) == 15
    assert f(3) == 15
    assert f.cache_info()["misses"] == 2


def test_a_class_execd_into_a_bare_dict_keys_what_its_classmethod_reads():
    """Such a class calls itself ``builtins``: it is the user's by its methods."""
    c = Cash(backend="memory", register_magic=False)
    ns = _namespace("bare")
    exec(RULES, ns)
    f = c.cache(ns["by_classmethod"])
    assert f(3) == 6
    _rebind(ns)
    assert f(3) == 15


@pytest.mark.parametrize("kind", ["named", "module"])
def test_unchanged_execd_code_still_hits(kind):
    c = Cash(backend="memory", register_magic=False)
    ns = _namespace(kind)
    exec(RULES, ns)
    f = c.cache(ns["by_method_helper"])
    assert [f(3) for _ in range(4)] == [6] * 4
    assert f.cache_info()["misses"] == 1


@pytest.fixture
def every_statement_cached(monkeypatch):
    monkeypatch.setenv("CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS", "0")
    monkeypatch.setenv("CASH_CALL_COST_FLOOR_SECONDS", "0")
    import cash.notebook.call_unit as call_unit

    monkeypatch.setattr(call_unit, "_COST_FLOOR_S", 0)


NOTEBOOK_CHANGES = [
    "ns['Config'].factor = 5",
    "ns['Config'].table[0] = 5",
    "ns['Config'].m = lambda self, x: x * 5",
    "exec('def helper(x):\\n    return x * 5\\n', ns)",
]
NOTEBOOK_CALLS = {
    "ns['Config'].factor = 5": "by_attr_helper",
    "ns['Config'].table[0] = 5": "by_table_helper",
    "ns['Config'].m = lambda self, x: x * 5": "by_method_helper",
    "exec('def helper(x):\\n    return x * 5\\n', ns)": "by_helper",
}


@pytest.mark.parametrize("edit", NOTEBOOK_CHANGES, ids=["rebind", "in_place", "method", "helper"])
def test_in_a_notebook_the_cell_after_the_change_reads_the_new_code(
    edit, every_statement_cached, cash_magics, mock_shell
):
    call = NOTEBOOK_CALLS[edit]
    cells = [
        f"ns = {{'__name__': 'rules'}}\nexec({RULES!r}, ns)",
        f"r1 = ns[{call!r}](3)",
        edit,
        f"r2 = ns[{call!r}](3)",
    ]
    for _run in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (mock_shell.user_ns["r1"], mock_shell.user_ns["r2"]) == (6, 15)
