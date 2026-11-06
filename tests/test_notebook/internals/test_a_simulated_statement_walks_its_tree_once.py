"""The analyses of a statement share one walk of its tree.

Each helper the simulation and the runtime run over a statement (what it
reads and writes, the aliases it binds, the streams it draws from, the
names it binds...) walked the statement's tree itself: simulating
``x = 1`` took some twenty walks of its four nodes, a third of what a
simulated statement cost. The trees are read, never changed, so
`walk_tree` walks each once and hands the same nodes to every helper.

Counted in ``ast.walk`` calls, not timed.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.ast_util import parse_cached
from cash.notebook.upstream.virtual_lineage import VirtualLineage
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls, tree_walks


@pytest.fixture(autouse=True)
def _memos_unchecked(monkeypatch):
    """The verify mode simulates each cell a memo stands in for as well."""
    monkeypatch.delenv("CASH_VERIFY_INCREMENTAL_SIM", raising=False)


def _notebook(n: int) -> list[str]:
    cells = ["base = 1", "reader = base + 1"]
    for i in range(n):
        cells.append(f"w{i} = {i}" if i % 2 == 0 else f"w{i} = [w{i - 1}] * 2\nv{i} = len(w{i})")
    cells.append("last = reader")
    return cells


def test_a_statement_simulated_again_is_not_walked_again(cash_magics, mock_shell):
    cells = _notebook(12)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    # Run a cell near the top: the next check below simulates it again,
    # with every statement of its text seen before.
    run_cash_cell(cash_magics, cells[1], cells=cells)

    with method_calls(VirtualLineage, "simulate_one_node") as simulated, tree_walks() as walks:
        run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert simulated.calls > 0
    # Before: 80 walks, the run of the cell's own statement counted too.
    assert walks.calls <= simulated.calls, f"{walks.calls} tree walks for {simulated.calls} simulated statements"
    assert mock_shell.user_ns["last"] == 2


def test_a_tree_is_walked_in_ast_walk_order_once():
    from cash.analysis.ast_util import walk_tree

    tree = parse_cached("a = [b + c for c in d]\nif a:\n    e = f(a, k=2)")

    with tree_walks() as walks:
        first = list(walk_tree(tree))
        again = list(walk_tree(tree))
        part = list(walk_tree(tree.body[1]))

    assert first == list(ast.walk(tree))
    assert again == first
    assert part == list(ast.walk(tree.body[1]))
    assert walks.calls == 2  # the tree, and the part asked for on its own
