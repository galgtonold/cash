"""Which cells holding IPython syntax cash runs itself, and how.

cash runs a cell with a magic line as IPython's transform writes it, so its
Python is tracked and runs once; a ``%%time``/``%%prun`` body runs under the
magic. A cell of magics alone, another cell magic, or a cell that loads an
extension or switches cash is left to IPython.
"""

from __future__ import annotations

import pytest
from IPython.core.inputtransformer2 import TransformerManager

from cash.analysis.cell_runs import jumpable_runs
from cash.analysis.code_analyzer import CodeAnalyzer, calls_ipython
from cash.notebook.ipython.ipython_cell import CellMagic, ipython_cell

TRANSFORM = TransformerManager().transform_cell


def test_a_magic_line_becomes_its_call_in_place():
    raw = "y = 1\n%time x = f()\nz = 2"
    cell = ipython_cell(raw, TRANSFORM)

    assert cell is not None and cell.magic is None
    assert [n.lineno for n in cell.tree.body] == [1, 2, 3]
    assert [calls_ipython(n) for n in cell.tree.body] == [False, True, False]
    assert cell.display_text(raw, cell.tree.body[1]) == "%time x = f()"
    assert cell.display_text(raw, cell.tree.body[0]) is None


@pytest.mark.parametrize(
    ("magic", "name", "line"), [("%%time", "time", ""), ("%%prun -q -s cumtime", "prun", "-q -s cumtime")]
)
def test_a_timed_or_profiled_body_runs_under_its_magic(magic, name, line):
    cell = ipython_cell(f"{magic}\nx = 1\n%matplotlib inline\ny = x", TRANSFORM)

    assert cell is not None
    assert cell.magic == CellMagic(name, line)
    assert [n.lineno for n in cell.tree.body] == [2, 3, 4]


@pytest.mark.parametrize(
    "raw",
    [
        "%matplotlib inline\nfiles = !ls",
        "%%capture out\nx = 1",
        "%%debug\nx = 1",
        "%%bash\necho hi",
        "%cash_off\nx = 1",
        "%load_ext autoreload\nx = 1",
        "x = = 1\n%time f()",
    ],
)
def test_ipython_runs_the_rest(raw):
    assert ipython_cell(raw, TRANSFORM) is None


def test_a_debugged_body_is_not_read_as_python():
    assert CodeAnalyzer.strip_magics("%%debug\nx = 1") == ""


def test_a_magic_assignment_ends_a_run_of_plain_assignments():
    tree = CodeAnalyzer.parse_cell(TRANSFORM("a = 1\nfiles = !ls\na = a + 1"))

    assert jumpable_runs(tree.body, "", lambda src: False) == {}
