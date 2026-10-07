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
