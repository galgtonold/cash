"""Inside a decomposed loop, a change through another name moves the whole.

``r = rows[i]; r.append(x)`` changes ``rows``; an inner loop changing the
outer loop's item changes what the outer loop walks. A later statement of the
pass reading the whole is keyed on the change.
"""

from __future__ import annotations

import ast

import pytest

from tests._cell_driver import run_cash_cell


def test_a_name_bound_to_a_part_of_another_is_found():
    from cash.notebook.statement.in_loop import part_bindings

    assert part_bindings(ast.parse("r = rows[i]")) == {"r": "rows"}
    assert part_bindings(ast.parse("w = model['w'].data")) == {"w": "model"}
    assert part_bindings(ast.parse("r = rows[i].copy()")) == {}
    assert part_bindings(ast.parse("r.append(1)")) == {}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize(
    "loop, after",
    [
        (
            "for r in rows:\n    for k in range(2):\n        r.append(k + lr)\n    s = sum(sum(x) for x in rows)\n    log.append(s)",
            [11, 16, 21],
        ),
        (
            "for i in range(len(rows)):\n    r = rows[i]\n    r.append(lr)\n    s = sum(sum(x) for x in rows)\n    log.append(s)",
            [8, 10, 12],
        ),
    ],
    ids=["nested", "by-index"],
)
def test_a_later_statement_of_the_pass_sees_the_change(cash_magics, mock_shell, loop, after):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["lr = 1", "rows = [[1], [2], [3]]\nlog = []", loop]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[0] = "lr = 2"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["log"] == after


def test_a_header_calling_an_unreadable_function_hands_out_every_mutable_global():
    from cash.analysis.mutation_effects import loop_target_roots

    records = [{"v": 1}]
    namespace = {"records": records, "rate": 2, "other": [1], "active": lambda: records}
    node = ast.parse("for r in active():\n    r['v'] = 1").body[0]
    roots = loop_target_roots(node, namespace, lambda name: False, lambda name: None)
    assert roots["r"] >= {"records", "other"}
    assert "rate" not in roots["r"]
