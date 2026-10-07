"""What a statement can reach is looked up live, from names read off its
text once.

The upstream scan asks every cell above for the environment it reads, which
follows the functions the cell calls (``reached_user_code``), on every cell:
re-walking each cell's tree made a trivial cell at the end of a 400-cell
notebook cost twice one at the top. The names a text reads never change; what
they are bound to is looked up each time.
"""

from __future__ import annotations

import ast
import types

import pytest

from cash.notebook import callee_reach


def test_the_tree_is_walked_once_per_text(monkeypatch):
    callee_reach.names_read.cache_clear()
    walked: list[object] = []
    real = ast.walk

    def counting(node):
        walked.append(node)
        return real(node)

    monkeypatch.setattr(callee_reach.ast, "walk", counting)
    ns = {"__name__": "__main__"}
    for _ in range(5):
        callee_reach.reached_user_code("y = helper(x) + mod.f(1)", ns)
    assert len(walked) == 1


def test_what_a_name_is_bound_to_is_looked_up_each_time():
    callee_reach.names_read.cache_clear()
    ns: dict = {"__name__": "__main__"}
    code = "y = helper(1)"
    assert callee_reach.reached_user_code(code, ns).functions == ()
    exec("def helper(v):\n    return v", ns)
    assert [f.__name__ for f in callee_reach.reached_user_code(code, ns).functions] == ["helper"]
    ns["helper"] = len
    assert callee_reach.reached_user_code(code, ns).functions == ()


def test_a_chain_through_a_module_is_followed():
    callee_reach.names_read.cache_clear()
    ns: dict = {"__name__": "__main__"}
    exec("def inner(v):\n    return v", ns)
    box = types.SimpleNamespace(inner=ns["inner"])
    ns["box"] = box
    # Only a module is followed along a chain: an object's attribute is not.
    assert callee_reach.reached_user_code("y = box.inner(1)", ns).functions == ()


def test_an_instance_of_a_c_type_is_not_walked_for_methods(monkeypatch):
    """An instance read is followed to its class's methods, but no cell
    gives an int or an array a method: walking ``vars(np.ndarray)`` for each
    one, for every cell above, grew a late cell of a long notebook by 60 ms."""
    np = pytest.importorskip("numpy")
    callee_reach.names_read.cache_clear()
    walked: list[object] = []
    real = callee_reach._class_member_functions

    def counting(member):
        walked.append(member)
        return real(member)

    monkeypatch.setattr(callee_reach, "_class_member_functions", counting)
    ns: dict = {"__name__": "__main__", "x": 3, "a": np.arange(3), "d": {"k": 1}}
    for _ in range(3):
        assert callee_reach.reached_user_code("y = x + a[1] + d['k']", ns).functions == ()
    assert walked == []


def test_an_instance_of_a_cells_class_is_still_followed_to_its_methods():
    """Control: a class a cell defines is no C type, so ``m.run()`` reaches
    ``Model.run``."""
    callee_reach.names_read.cache_clear()
    ns: dict = {"__name__": "__main__"}
    exec("class Model:\n    def run(self):\n        return 1\nm = Model()", ns)
    assert [f.__name__ for f in callee_reach.reached_user_code("y = m.run()", ns).functions] == ["run"]


def test_a_statement_of_values_and_library_calls_is_not_walked(monkeypatch):
    """``x2 = x1 + 1``, ``a = np.arange(9) * x``, ``print(x)``: the upstream
    scan asks every cell above about them on every cell; nothing they name
    leads to the user's code, which is told without setting up a walk."""
    import numpy as np

    callee_reach.names_read.cache_clear()
    ns: dict = {"__name__": "__main__", "np": np, "x1": 1, "a": np.arange(3), "d": {"k": 1}}
    callee_reach.reached_user_code("y = x1 + int(a[1]) + d['k']", ns)  # learns the C types
    walks: list[object] = []
    real = callee_reach._Found
    monkeypatch.setattr(callee_reach, "_Found", lambda namespace: walks.append(1) or real(namespace))
    for code in ("x2 = x1 + 1", "b = np.arange(9) * x1\nx3 = int(b[1])", "print(x1)", "e = d['k']"):
        assert callee_reach.reached_user_code(code, ns) == callee_reach._EMPTY
        assert walks == [], code


def test_a_function_or_local_module_is_walked_whatever_c_types_were_seen():
    """``types.FunctionType`` or ``type`` read by a statement are C types of
    no local module, and are remembered as such: a function or a module
    reached afterwards must still be followed, not taken for an instance of
    one."""
    callee_reach.names_read.cache_clear()
    ns: dict = {"__name__": "__main__", "types": types, "x": 1, "type": type}
    callee_reach.reached_user_code("k = (types.FunctionType, types.ModuleType, type(x), type)", ns)
    assert {types.FunctionType, types.ModuleType, type} <= callee_reach._FOREIGN_C_TYPES
    exec("def helper(v):\n    return v", ns)
    assert [f.__name__ for f in callee_reach.reached_user_code("y = helper(1)", ns).functions] == ["helper"]
