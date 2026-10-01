"""A function defined in a cell that also holds a magic is still found.

After a restart a helper the notebook defines is not live, so what it does
to its arguments is read from the cell text: the checker through
``NotebookSources``, the simulation through its own table of the notebook's
``def``s. Both parsed each cell as plain Python, so a cell such as
``%matplotlib inline`` above ``def add(d): d.append(1)`` did not parse and
its ``def`` was lost: ``add(items)`` then read as leaving ``items`` alone,
while the runtime, which has the live function, saw the mutation.
"""

from __future__ import annotations

from cash.analysis.mutation_effects import NotebookSources, cell_effects
from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream import NotebookSimulator

HELPER = "%matplotlib inline\ndef add(d):\n    d.append(1)"
CELLS = ["items = []", HELPER, "add(items)"]


def test_the_checker_sees_the_helper_mutate_its_argument():
    sources = NotebookSources(lambda: CELLS, "add(items)")
    assert "add" in sources.functions
    assert "items" in cell_effects("add(items)", sources, {}).mutated


def test_the_simulation_sees_the_helper_mutate_its_argument(mock_shell, cash_instance):
    simulation = NotebookSimulator(mock_shell, cash_instance, TrackingState()).virtual_lineage
    before_call = simulation.simulate(2, CELLS).virtual_lineage["items"]
    after_call = simulation.simulate(3, CELLS).virtual_lineage["items"]
    assert after_call != before_call, "add(items) left the lineage of items where it was"
