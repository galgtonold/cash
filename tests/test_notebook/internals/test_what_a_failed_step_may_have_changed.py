"""What a statement that raised may have changed, through the functions it
calls (`_changed_by_a_failed_step`): those names keep what the failed run
left instead of being rebuilt below it."""

from __future__ import annotations

import ast
import types

from cash.analysis.mutation_effects import live_function_source
from cash.notebook.ipython.cell_executor import _changed_by_a_failed_step


def _changed(code, ns):
    return _changed_by_a_failed_step(ast.parse(code).body[0], ns, lambda name: live_function_source(name, ns))


def _module(tmp_path, monkeypatch, source):
    (tmp_path / "failedproc.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "failedproc", raising=False)
    import failedproc

    return failedproc


def test_a_module_helper_that_fills_its_argument(tmp_path, monkeypatch):
    proc = _module(tmp_path, monkeypatch, "def fill(tot, data):\n    for k in data:\n        tot[k] = 1\n")
    ns = {"proc": proc, "tot": {}, "data": {"a": 1}}
    assert "tot" in _changed("proc.fill(tot, data)", ns)
    assert "data" not in _changed("proc.fill(tot, data)", ns)
    assert "tot" in _changed("for k in data:\n    proc.fill(tot, {k: 1})", ns)


def test_a_helper_that_only_reads_changes_nothing(tmp_path, monkeypatch):
    """The control: a helper that changes no argument leaves them alone."""
    proc = _module(tmp_path, monkeypatch, "def total(tot, data):\n    return sum(tot.values())\n")
    ns = {"proc": proc, "tot": {}, "data": {}}
    assert _changed("proc.total(tot, data)", ns) == set()


def test_a_callee_whose_source_cannot_be_read_may_change_what_it_is_handed():
    namespace: dict = {}
    exec("def opaque(a, b):\n    a.append(b)", namespace)  # no source to read back
    ns = {"opaque": types.FunctionType(namespace["opaque"].__code__, {}), "items": [], "x": 1}
    assert "items" in _changed("opaque(items, x)", ns)
