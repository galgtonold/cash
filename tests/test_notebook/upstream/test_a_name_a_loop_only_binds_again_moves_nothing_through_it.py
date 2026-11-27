"""A name a loop only binds again moves nothing through it.

Run a pass at a time and then as one unit, a loop moves on every name its
body binds, as what the loop made of it. Those moves were carried through
the names sharing an object with it too, as a change in place would be:
``actions = sorted(session['actions'], ...)`` moved ``sessions`` on though
the loop changed none of it. The simulation moves only what a structure
changes in place through those links, so the next cell found ``sessions``
built on an older value and rebuilt it. A list the loop does change, by
its text or through a library call, still moves what holds it.

The stream the library call draws from is seeded: unseeded, a shuffle that
happened to leave the first list as it was, stored on a slow run, was served
to every later pass as a draw frozen on its first value, and nothing moved.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    "body, moved",
    [
        ("actions = sorted(session['actions'], key=lambda a: a['at'])\n    count = len(actions)", False),
        ("actions = session['actions']\n    actions.append({'at': 0})", True),
        ("actions = session['actions']\n    random.shuffle(actions)", True),
    ],
    ids=["only bound again", "changed by its text", "changed by a library function"],
)
def test_a_loop_run_pass_by_pass_moves_what_it_changes_through_a_bound_name(cash_magics, body, moved):
    """Run a pass at a time, the names the body binds move on with the loop;
    one bound again and changed in no way moves nothing it shares an object
    with, as the simulation has it."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [
        "import random\nrandom.seed(0)\nsessions = [{'actions': [{'at': 3}, {'at': 1}, {'at': 2}]} for _ in range(50)]",
        f"for session in sessions:\n    {body}",
    ]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    lineage = cash_magics.tracking_state.variable_lineage["sessions"]

    run_cash_cell(cash_magics, cells[1], cells=cells)

    assert (cash_magics.tracking_state.variable_lineage["sessions"] != lineage) is moved
