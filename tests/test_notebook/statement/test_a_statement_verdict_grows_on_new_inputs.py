"""What a statement was first seen changing through a call is not final.

``np.copyto(a, src)`` with ``src`` equal to ``a`` changes nothing, and a
comprehension over no items calls nothing. When the statement runs again
under another key (new inputs), what its verdict does not hold is
fingerprinted again, and a change joins the verdict, the outputs and the
reasons it re-executes. A comprehension's element is watched as a loop
body's statement is.
"""

from __future__ import annotations

import ast
import heapq

import numpy as np
import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _run(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


@pytest.mark.usefixtures("store_everything")
def test_a_copy_of_equal_values_is_no_verdict_on_the_next(cash_magics, mock_shell):
    from cash.notebook.cache_key import statement_source_hash

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["import numpy as np\nsrc = np.zeros(3)", "a = np.zeros(3)", "np.copyto(a, src)", "r = float(a.sum())"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["r"] == 0.0
    assert cash_magics.tracking_state.mutation_verdicts[statement_source_hash("np.copyto(a, src)")] == set()
    cells[0] = "import numpy as np\nsrc = np.ones(3)"
    _run(cash_magics, cells)
    assert mock_shell.user_ns["r"] == 3.0
    assert "a" in cash_magics.tracking_state.mutation_verdicts[statement_source_hash("np.copyto(a, src)")]


@pytest.mark.usefixtures("store_everything")
def test_a_comprehension_handing_a_heap_to_a_library_is_watched(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["import heapq\npending = [7]", "h = []", "_ = [heapq.heappush(h, x) for x in pending]", "r = sorted(h)"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["r"] == [7]
    cells[0] = "import heapq\npending = [5, 3]"
    _run(cash_magics, cells)
    assert mock_shell.user_ns["r"] == [3, 5]


def test_what_a_comprehension_element_hands_a_library_is_a_candidate():
    from cash.analysis.mutation_effects import comprehension_watch_candidates

    namespace = {"heapq": heapq, "np": np, "h": [], "a": np.zeros(2), "xs": [1], "rows": [[2, 1]]}

    def candidates(code: str) -> frozenset[str]:
        return comprehension_watch_candidates(ast.parse(code), namespace, lambda name: name in {"heapq", "np"})

    assert "h" in candidates("_ = [heapq.heappush(h, x) for x in xs]")
    assert "h" in candidates("list(map(lambda x: heapq.heappush(h, x), xs))")
    assert "rows" in candidates("_ = {i: heapq.heapify(r) for i, r in enumerate(rows)}")
    assert candidates("m = [np.mean(a) for x in xs]") == frozenset()
    assert candidates("m = np.mean(a)") == frozenset()


@pytest.mark.usefixtures("store_everything")
def test_the_simulation_counts_what_the_verdict_lacks_once_inputs_moved(cash_magics, cash_instance, mock_shell):
    from cash.notebook.upstream import NotebookSimulator

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    _run(cash_magics, ["import numpy as np\nsrc = np.zeros(3)\na = np.zeros(3)", "np.copyto(a, src)"])
    statements = NotebookSimulator(mock_shell, cash_instance, cash_magics.tracking_state).virtual_lineage.statements
    held = dict(cash_magics.tracking_state.variable_lineage)
    _, _, same = statements.reads_writes("np.copyto(a, src)", ast.parse("np.copyto(a, src)"), set(), dict(held))
    assert "a" not in same
    moved = dict(held, src="another-lineage")
    _, _, outputs = statements.reads_writes("np.copyto(a, src)", ast.parse("np.copyto(a, src)"), set(), moved)
    assert "a" in outputs
