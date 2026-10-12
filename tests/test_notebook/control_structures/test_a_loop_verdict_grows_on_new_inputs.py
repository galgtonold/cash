"""What a loop was first seen changing through a library call is not final.

``while pending: heapq.heappush(h, pending.pop())`` first run on an empty
``pending`` changes nothing. When it runs again on items, what it hands to the
library is fingerprinted again and ``h`` joins the record, so the cell reading
``h`` is not served what it read before. On a state already observed nothing
is fingerprinted again. The simulation counts every such name for a loop whose
inputs changed since the runtime last ran it.
"""

from __future__ import annotations

import ast

import pytest

from tests._cell_driver import run_cash_cell

LOOP = "while pending:\n    heapq.heappush(h, pending.pop())"


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _run(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


@pytest.mark.usefixtures("store_everything")
def test_a_loop_first_run_on_no_items_learns_what_it_changes_later(cash_magics, mock_shell):
    from cash.notebook.loop_watch import loop_watch_hash

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["import heapq\npending = []", "h = []", LOOP, "r = sorted(h)"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["r"] == []
    assert cash_magics.tracking_state.mutation_verdicts[loop_watch_hash(LOOP)] == set()

    cells[0] = "import heapq\npending = [5, 3, 9]"
    _run(cash_magics, cells)
    assert mock_shell.user_ns["r"] == [3, 5, 9]
    assert {"h", "pending"} <= cash_magics.tracking_state.mutation_verdicts[loop_watch_hash(LOOP)]


def test_a_known_verdict_fingerprints_again_only_what_it_does_not_hold():
    from cash.notebook.loop_watch import LoopWatch

    namespace = {"h": [], "pending": [1]}
    watch = LoopWatch.start(LOOP, frozenset({"h", "pending"}), {"pending"}, namespace, rewatch=True)
    assert set(watch.before) == {"h"}
    namespace["h"].append(1)
    assert watch.finish(namespace) == {"h", "pending"}
    settled = LoopWatch.start(LOOP, frozenset({"h", "pending"}), {"pending"}, namespace)
    assert not settled.watching
    assert settled.finish(namespace) == {"pending"}


@pytest.mark.usefixtures("store_everything")
def test_the_simulation_counts_every_candidate_once_inputs_changed(cash_magics, cash_instance, mock_shell):
    from cash.notebook.upstream import NotebookSimulator

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    _run(cash_magics, ["import heapq\npending = []\nh = []", LOOP])
    simulation = NotebookSimulator(mock_shell, cash_instance, cash_magics.tracking_state).virtual_lineage
    node = ast.parse(LOOP).body[0]
    assert "h" not in simulation.controls.collect_mutations(node, set(), set(), set(), lambda: False)
    assert "h" in simulation.controls.collect_mutations(node, set(), set(), set(), lambda: True)
