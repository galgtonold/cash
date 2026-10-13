"""``nx.set_node_attributes(G, ...)`` changes ``G``, not the module ``nx``.

A module function named ``set_*`` counts as changing a setting the module
keeps (``pd.set_option``), so the module's lineage moves with it. networkx's
``set_node_attributes(G, values, name)`` was counted that way too: every
re-run of the cell moved the lineage of ``nx``, so every statement reading
``nx`` keyed apart from the runs before, and after a restart -- which ran
the cell fewer times -- betweenness, Louvain and label propagation on an
unchanged graph all ran again (about 150 s).

A ``set_*`` call handed names that is seen changing one of them sets on that
name; the module keeps its lineage. One that changes none of them (``set_option(k, 3)``)
still counts as a module setting, as does one handed no name at all.
"""

from __future__ import annotations

import ast
import os
import sys
import types

import pytest

from cash.analysis.mutation_effects import classify_receivers

LIB = (
    "class Graph:\n"
    "    def __init__(self, n):\n"
    "        self.attrs = {i: {} for i in range(n)}\n"
    "def make(n):\n    return Graph(n)\n"
    "def set_node_attributes(g, values, name):\n"
    "    for k, v in values.items():\n"
    "        g.attrs[k][name] = v\n"
    "OPTIONS = {}\n"
    "def set_option(key, value):\n    OPTIONS[key] = value\n"
)


@pytest.fixture
def graphlib(tmp_path, monkeypatch):
    name = f"_graphlib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


def _lineage(magics, name):
    return magics.tracking_state.variable_lineage.get(name)


def test_the_setter_moves_the_graph_not_the_module(cash_magics, statement_processor, graphlib):
    statement_processor.process_statement(f"import {graphlib} as gl")
    statement_processor.process_statement("g = gl.make(5)")
    module, graph = _lineage(cash_magics, "gl"), _lineage(cash_magics, "g")

    metrics = statement_processor.process_statement("gl.set_node_attributes(g, {0: 'a', 1: 'b'}, 'region')")

    reasons = " ".join(metrics.get("uncacheable_reasons") or ())
    assert "In-place mutation on: g" in reasons, reasons
    assert "gl" not in reasons.replace("In-place mutation on: g ", ""), reasons
    assert _lineage(cash_magics, "gl") == module
    assert _lineage(cash_magics, "g") != graph
    assert sys.modules[graphlib].make  # the module is untouched


def test_running_the_setter_again_leaves_the_module_alone(cash_magics, statement_processor, graphlib):
    statement_processor.process_statement(f"import {graphlib} as gl")
    module = _lineage(cash_magics, "gl")
    for _ in range(3):
        statement_processor.process_statement("g = gl.make(5)")
        statement_processor.process_statement("gl.set_node_attributes(g, {0: 'a'}, 'region')")
    assert _lineage(cash_magics, "gl") == module


def test_a_setter_that_changes_none_of_its_names_sets_the_module(cash_magics, statement_processor, graphlib):
    statement_processor.process_statement(f"import {graphlib} as gl")
    statement_processor.process_statement("k = 'width'")
    module = _lineage(cash_magics, "gl")

    metrics = statement_processor.process_statement("gl.set_option(k, 3)")

    assert "gl" in " ".join(metrics.get("uncacheable_reasons") or ())
    assert _lineage(cash_magics, "gl") != module
    assert sys.modules[graphlib].OPTIONS == {"width": 3}


# The rule the runtime and the simulation share.

_NS = {"nx": types.ModuleType("networkx"), "pd": types.ModuleType("pandas"), "G": object(), "w": 3}


def _classes(code, verdict):
    tree = ast.parse(code)
    return classify_receivers(tree, _NS, lambda: verdict, arguments=frozenset({"G", "w"}))


def test_no_verdict_yet_counts_the_module_until_the_run_is_seen():
    classes = _classes("nx.set_node_attributes(G, w, 'x')", None)
    assert "nx" in classes.mutated
    assert classes.provisional == {"nx": frozenset({"G", "w"})}


def test_a_verdict_naming_a_handed_name_leaves_the_module_out():
    classes = _classes("nx.set_node_attributes(G, w, 'x')", {"G"})
    assert "nx" not in classes.mutated
    assert "G" in classes.mutated


def test_a_verdict_naming_none_of_them_keeps_the_module():
    assert "nx" in _classes("nx.set_node_attributes(G, w, 'x')", set()).mutated


@pytest.mark.parametrize("code", ["pd.set_option('display.width', 80)", "pd.set_option('a', w)\npd.set_option('b', 1)"])
def test_a_setting_handed_no_name_is_always_the_module_s(code):
    classes = _classes(code, {"w"})
    assert "pd" in classes.mutated
    assert not classes.provisional
