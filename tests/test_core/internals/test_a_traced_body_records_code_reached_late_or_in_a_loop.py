"""What a traced body records stays whole while each code is seen once.

A cached call's body runs under ``sys.monitoring`` (`cash.decorator.code_trace`):
each code object of the user's costs one callback per call, then its event is
disabled. Code reached only deep inside a hot loop, or reached later through a
second call path, is still in the trace; and a generator expression's closure
cells, which no check reads, are not digested as it starts.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash.decorator import code_trace

_PLUGINS = """
def common(x):
    return x + 1


def rare(x):
    return x * 2


def late(x):
    return x - 1
"""


@pytest.fixture
def load(tmp_path, request):
    """Write ``plugins_<test>.py`` and a module *body* importing it as
    ``plugins``, and import the module."""
    tag = "".join(c if c.isalnum() else "_" for c in request.node.name)
    plugins, module = f"plugins_{tag}", f"m_{tag}"
    sys.path.insert(0, str(tmp_path))

    def _load(body: str, plugin_source: str = _PLUGINS):
        for name in (plugins, module):
            sys.modules.pop(name, None)
        (tmp_path / f"{plugins}.py").write_text(textwrap.dedent(plugin_source), encoding="utf-8")
        (tmp_path / f"{module}.py").write_text(
            f"import cash\nimport {plugins} as plugins\nRUNS = []\n" + textwrap.dedent(body), encoding="utf-8"
        )
        importlib.invalidate_caches()
        return importlib.import_module(module)

    yield _load
    sys.path.remove(str(tmp_path))
    for name in (plugins, module):
        sys.modules.pop(name, None)


def _recorded(fn, *args):
    """Run *fn* as a traced body and return its unit."""
    unit = code_trace.begin()
    assert unit is not None, code_trace.recorder_or_reason().reason
    try:
        fn(*args)
    finally:
        code_trace.end(unit)
    return unit


_LOOP = """
def body(n):
    total = 0
    for i in range(n):
        total = plugins.common(total)
        if i == n - 1:
            total = plugins.rare(total)
    return total


def first(x):
    return plugins.common(x)


def second(x):
    return plugins.late(x)


def two_paths(n):
    total = 0
    for _ in range(n):
        total = first(total)
    # Reached only after the loop, through another caller.
    return second(total)
"""


def test_a_function_reached_only_on_a_hot_loops_last_turn_is_recorded(load):
    m = load(_LOOP)
    unit = _recorded(m.body, 5000)
    codes = set(unit.codes)
    assert m.plugins.common.__code__ in codes
    assert m.plugins.rare.__code__ in codes


def test_a_second_call_path_reached_later_in_the_body_is_recorded(load):
    m = load(_LOOP)
    unit = _recorded(m.two_paths, 3000)
    codes = set(unit.codes)
    assert {m.first.__code__, m.second.__code__, m.plugins.common.__code__, m.plugins.late.__code__} <= codes


def test_each_call_records_again_what_the_one_before_disabled(load):
    m = load(_LOOP)
    _recorded(m.body, 50)
    unit = _recorded(m.two_paths, 50)
    assert m.plugins.common.__code__ in unit.codes
    assert m.plugins.late.__code__ in unit.codes


_CLOSURES = """
class Box:
    def __init__(self):
        self.items = list(range(50))

    def total(self):
        return sum(x for x in self.items if x != id(self))


def make(box):
    return box.total(), (lambda: box)()
"""


def test_a_generator_expressions_cells_are_not_digested_but_a_lambdas_are(load):
    m = load(_CLOSURES)
    unit = _recorded(m.make, m.Box())
    genexpr = next(c for c in unit.codes if c.co_name == "<genexpr>")
    lam = next(c for c in unit.codes if c.co_name == "<lambda>")
    assert genexpr.co_freevars == ("self",)
    assert genexpr not in unit.cells
    assert lam in unit.cells


_BY_NAME_LOOP = """
@cash.cache
def run(name, n):
    RUNS.append(name)  # @cash:assume-safe
    total = 0
    for i in range(n):
        total = plugins.common(total)
        if i == n - 1:
            total = getattr(plugins, name)(total)
    return total
"""


def test_an_edit_to_code_reached_only_on_a_loops_last_turn_runs_the_call_again(load):
    m = load(_BY_NAME_LOOP)
    assert m.run("rare", 2000) == 4000
    assert m.run("rare", 2000) == 4000
    assert m.RUNS == ["rare"]
    m.plugins.rare.__code__ = (lambda x: x * 3).__code__
    assert m.run("rare", 2000) == 6000
    assert m.RUNS == ["rare", "rare"]
