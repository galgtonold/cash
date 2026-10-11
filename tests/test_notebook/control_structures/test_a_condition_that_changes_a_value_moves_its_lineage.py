"""What the condition of an ``if`` changes is a change, whichever branch runs.

``if opts.pop('debug', False):`` removes a key, and ``if stack.pop() > 7:``
in a loop shortens the list on every pass. Only what the branches change got
a new lineage, so a statement reading the dict or list after the ``if`` was
keyed as before it and served the result from before the change: on the
first run, the same statement above the ``if`` handed its result to the one
below it.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.mutation_effects import control_structure_mutations
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"import time\ndef describe(v):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return sorted(v)\n"


@pytest.mark.parametrize(
    ("test", "changed"),
    [
        ("opts.pop('debug', False)", {"opts"}),
        ("stack.pop() > 7", {"stack"}),
        ("not seen.add(k)", {"seen"}),
        ("d.setdefault(k, []) and q", {"d"}),
        # The result is used: a method not known to change its receiver
        # does not, as in `_ = name.startswith('a')`.
        ("cfg.get('debug')", set()),
        ("name.startswith('a')", set()),
        ("path.exists()", set()),
        ("len(stack) > 3", set()),
    ],
)
def test_condition_mutations(test, changed):
    from cash.analysis.mutation_effects import condition_mutations  # noqa: PLC0415 - the name the fix adds

    assert condition_mutations(ast.parse(test, mode="eval").body) == changed


def test_every_condition_of_a_chain_counts():
    node = ast.parse("if a.pop():\n    pass\nelif b.pop():\n    pass\nelse:\n    pass").body[0]
    assert control_structure_mutations(node, lambda n: False, lambda n: False) == {"a", "b"}


def test_a_statement_after_the_if_sees_what_its_condition_removed(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, SLOW + "opts = {'lr': 0.1, 'epochs': 3, 'debug': True}")
    run_cash_cell(cash_magics, "keys = describe(opts)")
    assert ns["keys"] == ["debug", "epochs", "lr"]
    run_cash_cell(cash_magics, "if opts.pop('debug', False):\n    mode = 'debug'")
    run_cash_cell(cash_magics, "keys = describe(opts)")
    assert ns["keys"] == ["epochs", "lr"]


def test_a_condition_that_takes_no_branch_still_moves_the_lineage(cash_magics):
    lineage = cash_magics.tracking_state.variable_lineage
    run_cash_cell(cash_magics, "opts = {'lr': 0.1}")
    before = lineage["opts"]
    run_cash_cell(cash_magics, "if opts.pop('lr', None) is None:\n    mode = 'none'")
    assert cash_magics.shell.user_ns["opts"] == {}
    assert lineage["opts"] != before


def test_a_loop_whose_condition_pops_moves_the_list_on_every_pass(cash_magics):
    # Every pass has the same loop value, so only the list's lineage tells
    # the passes' readers apart; no branch is taken after the second pass.
    ns = cash_magics.shell.user_ns
    lineage = cash_magics.tracking_state.variable_lineage
    loop = "seen = []\nfor i in [0] * 5:\n    if stack.pop() > 7:\n        pass\n    top = describe(stack)[-1]\n    seen.append(top)"
    for _ in range(2):
        run_cash_cell(cash_magics, SLOW + "stack = list(range(10))")
        before = lineage["stack"]
        run_cash_cell(cash_magics, loop)
        assert ns["seen"] == [8, 7, 6, 5, 4]
        assert lineage["stack"] != before
    run_cash_cell(cash_magics, SLOW + "stack = list(range(10))")
    run_cash_cell(cash_magics, "after = describe(stack)")
    assert ns["after"] == list(range(10))


def test_a_condition_that_changes_nothing_reads_nothing(cash_magics):
    lineage = cash_magics.tracking_state.variable_lineage
    run_cash_cell(cash_magics, "cfg = {'debug': False}\nhits = []")
    before = lineage["cfg"]
    run_cash_cell(cash_magics, "for i in range(5):\n    if cfg.get('debug'):\n        hits.append(i)")
    assert lineage["cfg"] == before
