"""``cash.analysis.mutation_effects``: the one analysis entry point the upstream
checker, the statement processor and the upstream simulation ask what a cell
or a statement writes."""

from __future__ import annotations

import ast
import os

from cash.analysis.mutation_effects import (
    CellEffects,
    NotebookSources,
    cell_effects,
    is_module_name,
    statement_effects,
)


def _effects(cell: str, *others: str, namespace=None) -> CellEffects:
    return cell_effects(cell, NotebookSources(lambda: [*others, cell], cell), namespace or {})


class TestNotebookSources:
    def test_the_last_definition_wins(self):
        sources = NotebookSources(lambda: ["def f():\n    return 1", "def f():\n    return 2"], "x = f()")
        assert sources.functions["f"].endswith("return 2")

    def test_the_current_cell_counts_even_when_unsaved(self):
        sources = NotebookSources(lambda: [], "class K:\n    pass\nk = K()")
        assert set(sources.classes) == {"K"}
        assert sources.instance_class("k") == "K"

    def test_the_notebook_is_read_only_when_asked(self):
        calls = []

        def cells():
            calls.append(1)
            return []

        assert _effects_with_loader("x = 1", cells) == CellEffects(
            outputs=frozenset({"x"}), reassigned=frozenset({"x"})
        )
        assert calls == []

    def test_aliases_are_followed_to_the_factory(self):
        sources = NotebookSources(
            lambda: ["def make():\n    n = 0\n    def inc():\n        nonlocal n\n        n += 1\n    return inc"],
            "c = make()\nd = c",
        )
        assert sources.factory_def("d") is None
        assert sources.factory_def("d", follow_aliases=True).name == "make"

    def test_partial_bindings(self):
        sources = NotebookSources(lambda: ["import functools\np = functools.partial(f, acc, 3)"], "")
        assert sources.partial_bindings == {"p": ("f", ["acc", None])}


def _effects_with_loader(cell, loader):
    return cell_effects(cell, NotebookSources(loader, cell), {})


class TestCellEffects:
    def test_a_magic_cell_writes_nothing_it_can_see(self):
        assert _effects("%time x = 1") == CellEffects()

    def test_a_callee_global_is_mutated_and_an_input_but_not_an_output(self):
        effects = _effects("r = bump()", "G = []\ndef bump():\n    G.append(1)")
        assert "G" in effects.mutated
        assert effects.hidden_inputs == {"G"}
        assert "G" not in effects.outputs

    def test_no_cache_writes_are_left_to_accumulate(self):
        effects = _effects("n = n + 1  # @cash: no-cache\nlst.append(n)")
        assert effects.nocache == {"n"}
        assert "n" not in effects.reassigned
        assert effects.method_receivers == {"lst"}

    def test_channels(self):
        effects = _effects("df['a'] = df['a'] * 2\na, b = b, a\nadd(data)", "def add(d):\n    d.append(1)")
        assert effects.selfref == {"df"}
        assert effects.crossref_reassigned == {"a", "b"}
        assert "data" in effects.method_receivers and "data" in effects.mutated

    def test_state_on_a_function_object(self):
        effects = _effects("collect(1)", "def collect(x, acc=[]):\n    acc.append(x)")
        assert effects.stateful_funcs == {"collect"}

    def test_a_receiver_mutated_by_another_cell_too_is_not_reset(self):
        klass = "class Dag:\n    def __init__(self):\n        self.edges = []\n    def link(self, e):\n        self.edges.append(e)"
        sole = _effects("g = Dag()\ng.link(1)", klass)
        shared = _effects("g = Dag()\ng.link(1)", klass, "g.link(2)")
        assert "g" in sole.method_receivers
        assert "g" not in shared.method_receivers

    def test_a_numpy_view_mutates_its_base(self):
        class ndarray:  # noqa: N801 - duck-types numpy's type name
            __module__ = "numpy"

        effects = _effects("v = arr[1:]\nv += 1", namespace={"arr": ndarray()})
        assert "arr" in effects.mutated
        assert "arr" not in _effects("v = arr[1:]\nv += 1", namespace={"arr": [1, 2]}).mutated


class TestStatementEffects:
    SRCS = {
        "compute": "def compute(v):\n    CALLS.append(v)\n    return v",
        "fill": "def fill(d):\n    d.append(1)",
    }

    def _f(self, code, namespace, **kwargs):
        return statement_effects(code, ast.parse(code), namespace=namespace, resolve_source=self.SRCS.get, **kwargs)

    def test_a_callee_global_is_written(self):
        effects = self._f("x = compute(1)", {"CALLS": []})
        assert effects.callee_globals == {"CALLS"}
        assert effects.outputs == {"CALLS", "x"}
        # Not bound in the namespace: the callee's own global, not a variable.
        assert self._f("x = compute(1)", {}).callee_globals == frozenset()

    def test_a_control_body_leaves_callee_globals_to_its_loop(self):
        assert self._f("x = compute(1)", {"CALLS": []}, control_body=True).callee_globals == frozenset()

    def test_an_argument_a_callee_mutates(self):
        assert self._f("fill(data)", {"data": []}).arg_mutations == {"data"}
        assert self._f("fill(os)", {"os": os}).arg_mutations == frozenset()
        assert self._f("fill(np)", {}, virtual_modules={"np"}).arg_mutations == frozenset()

    def test_an_unparsed_statement_still_has_inputs_and_outputs(self):
        effects = statement_effects("y = await f(x)", None, namespace={}, resolve_source=self.SRCS.get)
        assert effects.outputs == {"y"} and {"f", "x"} <= effects.inputs


def test_is_module_name():
    assert is_module_name("os", {"os": os})
    assert not is_module_name("os", {"os": 1}, {"os"})
    assert is_module_name("np", {}, {"np"})
    assert not is_module_name("np", {})
