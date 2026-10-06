"""What the upstream check reads off a cell's text is worked out once per text.

Every cell asks the check about every cell above it, so a trivial cell at the
end of a 400-cell notebook cost twice one at the top: each ``def`` above was
unparsed again (twice: for the cell's effects and for the simulation), and
each statement of the trace parsed and unparsed again for its codes.
"""

from __future__ import annotations

import ast

from cash.analysis import mutation_effects
from cash.analysis.mutation_effects import NotebookSources
from cash.notebook.upstream import virtual_lineage

CELLS = [f"def h{i}(v):\n    return v + {i}\nx{i} = h{i}(1)" for i in range(30)]


def _count_unparse(monkeypatch) -> list:
    calls: list = []
    real = ast.unparse

    def counting(node):
        calls.append(node)
        return real(node)

    monkeypatch.setattr(ast, "unparse", counting)
    return calls


def test_definitions_are_unparsed_once(monkeypatch):
    mutation_effects.cell_definitions.cache_clear()
    calls = _count_unparse(monkeypatch)
    first = NotebookSources(lambda: CELLS, "y = 1").functions
    for current in ("y = 2", "y = 3", "y = 4"):
        assert NotebookSources(lambda: CELLS, current).functions == first
    assert len(calls) == len(CELLS)
    assert first["h7"].startswith("def h7(v):")


def test_a_later_definition_still_wins():
    cells = ["def f():\n    return 1", "def f():\n    return 2"]
    assert NotebookSources(lambda: cells, "y = f()").functions["f"].endswith("return 2")


def test_trace_codes_are_worked_out_once(monkeypatch):
    virtual_lineage._trace_codes.cache_clear()
    calls = _count_unparse(monkeypatch)
    loop = "for i in range(3):\n    total = total + i\n    print(i)"
    for _ in range(4):
        assert virtual_lineage._trace_codes(loop) == (loop, "total = total + i", "print(i)")
    assert len(calls) == 2
