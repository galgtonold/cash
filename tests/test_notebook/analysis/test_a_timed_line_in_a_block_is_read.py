"""A ``%time`` line in a loop or ``if`` body is read for what its Python does.

``for i in range(2): %time acc.append(i * k)`` changes ``acc`` and reads
``k`` as the plain loop does. The analysis read only the magic's call, whose
Python is a string, so the loop changed nothing it could see: after an edit
of ``k`` above, even a full Run All served the cell below from the cache with
the list the old ``k`` left. A cell holding nothing but such a loop was not
run through cash at all.
"""

from __future__ import annotations

import ast

import pytest
from IPython.core.inputtransformer2 import TransformerManager

from cash.analysis.code_analyzer import CodeAnalyzer
from cash.analysis.mutation_effects import control_structure_mutations
from cash.notebook.ipython.ipython_cell import ipython_cell

TRANSFORM = TransformerManager().transform_cell

LOOP = "for i in range(2):\n    %time acc.append(i * k)"
BRANCH = "if k:\n    %time acc.append(k)"


def _tree(cell: str) -> tuple[str, ast.Module]:
    source = TRANSFORM(cell)
    return source, ast.parse(source)


@pytest.mark.parametrize("cell", [LOOP, BRANCH, "%timeit -n1 -r1 acc.append(k)"])
def test_the_timed_python_is_read(cell):
    source, tree = _tree(cell)
    inputs, _ = CodeAnalyzer.analyze_code_block(source, tree=tree)
    assert {"acc", "k"} <= inputs


def test_what_the_timed_python_binds_is_written():
    source, tree = _tree("for i in range(2):\n    %time last = i * k")
    _, outputs = CodeAnalyzer.analyze_code_block(source, tree=tree)
    assert "last" in outputs


@pytest.mark.parametrize("cell", [LOOP, BRANCH])
def test_the_structure_changes_what_the_timed_python_changes(cell):
    _, tree = _tree(cell)
    assert control_structure_mutations(tree.body[0], lambda _n: False, lambda _n: False) == {"acc"}


def test_a_shell_command_in_a_body_is_not_read_as_python():
    _, tree = _tree("for i in range(2):\n    !echo hi")
    assert control_structure_mutations(tree.body[0], lambda _n: False, lambda _n: False) == set()


def test_a_cell_holding_only_such_a_loop_is_run_by_cash():
    cell = ipython_cell(LOOP, TRANSFORM)
    assert cell is not None and isinstance(cell.tree.body[0], ast.For)


def test_a_cell_of_magic_lines_only_is_still_left_to_ipython():
    """Control: nothing in it for cash to track."""
    assert ipython_cell("%time f()\nfiles = !ls", TRANSFORM) is None
