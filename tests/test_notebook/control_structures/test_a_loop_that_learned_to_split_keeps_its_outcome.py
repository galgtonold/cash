"""A loop that learned a split is modelled by the outcome of its last run.

A split verdict is recorded at the end of a run that ran the loop whole, and
applies from the next run. The simulation modelled such a loop as its head and
tail at once, and neither half had an outcome of its own, so what the loop
built was given a lineage from the formula, not the one the runtime recorded:
every entry built on it missed, and after a restart the loop and everything
above it ran again. The integration twin, with the restart, is
``tests/test_notebook_integration/restart/test_restart_loop_outcomes.py``.
"""

import ast

from cash.notebook.loop_split import loop_source_hash, store_for_backend
from tests._cell_driver import run_cash_cell

SEED = "parts = []"
LOOP = "for i in range(40):\n    parts.append(i * 2)"
CELLS = [SEED, LOOP, "total = sum(parts)"]


def _learning_to_split(cash_instance) -> None:
    # The verdict is a wall-clock judgement of the first iterations; raising
    # its ceiling and dropping its floor makes every run record one.
    cash_instance.config.loop_split_max_iter_seconds = 1.0
    cash_instance.config.loop_split_min_remaining_seconds = 0.0


def test_the_simulation_gives_what_the_loop_built_its_recorded_lineage(cash_magics, cash_instance):
    _learning_to_split(cash_instance)
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    loop = ast.parse(LOOP).body[0]
    assert store_for_backend(cash_instance.backend).get(loop_source_hash(loop)) is not None, "no split was learned"
    built = cash_magics.tracking_state.variable_lineage["parts"]

    run_cash_cell(cash_magics, CELLS[2], cells=CELLS)

    assert cash_magics.tracking_state.simulated_lineage.get("parts") == built, (
        "the simulation modelled the loop as the head and tail it has never run as, "
        "so `parts` got a lineage no entry was written with"
    )


def test_a_changed_input_still_models_the_split(cash_magics, cash_instance):
    """The recorded outcome is used only while it holds; otherwise the halves
    are what the planner will run, and what the simulation must model."""
    _learning_to_split(cash_instance)
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    built = cash_magics.tracking_state.variable_lineage["parts"]
    edited = ["parts = [99]", LOOP, "total = sum(parts)"]

    run_cash_cell(cash_magics, edited[2], cells=edited)

    assert cash_magics.tracking_state.simulated_lineage.get("parts") not in (None, built)
