"""One callee resolver, which never runs the user's code.

The decorator's analyzer, the notebook's cacheability scan and the upstream
simulation each resolved a call's callee their own way. The simulation's
followed ``getattr`` on any object, so simulating ``x = cfg.value()`` read a
property of the user's -- in a pass meant to have no effects.
"""

from __future__ import annotations

import ast
import os.path
import types

import pytest

from cash.analysis.ast_util import called_names, resolve_callee
from cash.notebook.upstream.simulator import _binds_without_reading


class Loud:
    """Every way an attribute read can run code, counted."""

    def __init__(self):
        self.reads = 0
        self.stored = len

    @property
    def value(self):
        self.reads += 1
        return lambda: 1

    def __getattr__(self, name):
        self.reads += 1
        return lambda: 1

    def method(self):
        return 1

    @staticmethod
    def static():
        return 2

    @classmethod
    def build(cls):
        return cls


def _func(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def test_a_name_and_a_module_path_resolve():
    ns = {"os": os, "f": print}
    assert resolve_callee(_func("f"), ns) is print
    assert resolve_callee(_func("os.path.join"), ns) is os.path.join
    assert resolve_callee(_func("missing.x"), ns) is None
    assert resolve_callee(_func("f()"), ns) is None
    assert resolve_callee(_func("xs[0]"), {"xs": [print]}) is None


def test_builtins_only_when_asked():
    assert resolve_callee(_func("sorted"), {}) is None
    assert resolve_callee(_func("sorted"), {}, builtins_fallback=True) is sorted


@pytest.mark.parametrize("modules_only", [True, False])
@pytest.mark.parametrize("expr", ["obj.value", "obj.nothing_here", "obj.value.inner"])
def test_no_property_or_getattr_runs(expr, modules_only):
    obj = Loud()
    assert resolve_callee(_func(expr), {"obj": obj}, modules_only=modules_only) is None
    assert obj.reads == 0


def test_modules_only_stops_at_an_object():
    obj = Loud()
    assert resolve_callee(_func("obj.method"), {"obj": obj}) is None


def test_methods_bind_as_python_binds_them():
    obj = Loud()
    ns = {"obj": obj, "Loud": Loud, "rows": []}
    bound = resolve_callee(_func("obj.method"), ns, modules_only=False)
    assert isinstance(bound, types.MethodType) and bound.__self__ is obj and bound.__func__ is Loud.method
    assert resolve_callee(_func("Loud.method"), ns, modules_only=False) is Loud.method
    assert resolve_callee(_func("obj.static"), ns, modules_only=False) is Loud.__dict__["static"].__func__
    assert resolve_callee(_func("obj.build"), ns, modules_only=False)() is Loud
    assert resolve_callee(_func("obj.stored"), ns, modules_only=False) is len
    append = resolve_callee(_func("rows.append"), ns, modules_only=False)
    assert append is not None and append.__self__ is ns["rows"]
    assert obj.reads == 0


def test_the_simulation_reads_no_property():
    obj = Loud()
    assert _binds_without_reading("x = cfg.value()", {"cfg": obj}) is False
    assert obj.reads == 0
    assert _binds_without_reading("p = os.path.join('a', 'b')", {"os": os}) is True


def test_a_mock_standing_in_for_a_module_is_read():
    """`mock.patch.object(app, "json", MagicMock())`: the analyzer must see the
    mock's `loads` to refuse caching what it returns."""
    from unittest import mock

    fake = mock.MagicMock()
    found = resolve_callee(_func("json.loads"), {"json": fake})
    assert found is fake.loads


_SCOPES_CELL = """
setup()
def helper():
    inner()
class Box:
    made = build()
if flag():
    in_if()
    def nested():
        deep()
for t in source():
    per_item(t)
f = lambda: later()
"""


def test_called_names_scopes():
    tree = ast.parse(_SCOPES_CELL)
    everything = {"setup", "inner", "build", "flag", "in_if", "deep", "source", "per_item", "later"}
    assert called_names(tree) == everything
    # The cell's own def/class statements are skipped, a def inside an `if` is not.
    assert called_names(tree, "top_level") == everything - {"inner", "build"}
    # Only what runs while the cell itself runs.
    assert called_names(tree, "eager") == {"setup", "flag", "in_if", "source", "per_item"}
    # A control structure's header belongs to the cell, its body does not.
    assert called_names(tree, "no_control_bodies") == {"setup", "inner", "build", "flag", "source", "later"}
    assert called_names(None) == frozenset()
