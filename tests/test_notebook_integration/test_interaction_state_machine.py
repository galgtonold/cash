"""Batch 216 – State machine interaction tests.

Tests editing cells that implement state machine logic
with transitions and verifying correct propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestStateMachineEdits:
    """Editing state machine patterns."""

    def test_edit_transition_table(self, nb_runner):
        """Edit state transition rules."""
        nb_runner.create_notebook([
            "transitions = {'idle': 'running', 'running': 'done', 'done': 'idle'}",
            "state = 'idle'\nfor _ in range(3):\n    state = transitions[state]\nprint(f'state = {state}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "state = idle" in nb_runner.get_output(2)

        # Change transitions
        nb_runner.set_cell_source(1, "transitions = {'idle': 'running', 'running': 'paused', 'paused': 'running'}")
        nb_runner.run_all()
        assert "state = running" in nb_runner.get_output(2)

    def test_edit_initial_state(self, nb_runner):
        """Edit starting state."""
        nb_runner.create_notebook([
            "rules = {'A': 'B', 'B': 'C', 'C': 'A'}",
            "current = 'A'\npath = [current]\nfor _ in range(4):\n    current = rules[current]\n    path.append(current)\nprint(f'path = {path}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "path = ['A', 'B', 'C', 'A', 'B']" in nb_runner.get_output(2)

        # Change start
        nb_runner.set_cell_source(2, "current = 'C'\npath = [current]\nfor _ in range(4):\n    current = rules[current]\n    path.append(current)\nprint(f'path = {path}')")
        nb_runner.run_all()
        assert "path = ['C', 'A', 'B', 'C', 'A']" in nb_runner.get_output(2)

    def test_edit_event_sequence(self, nb_runner):
        """Edit event sequence processing."""
        nb_runner.create_notebook([
            "events = ['start', 'pause', 'resume', 'stop']",
            "log = []\nfor e in events:\n    log.append(e.upper())\nprint(f'log = {log}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log = ['START', 'PAUSE', 'RESUME', 'STOP']" in nb_runner.get_output(2)

        # Change events
        nb_runner.set_cell_source(1, "events = ['init', 'process', 'complete']")
        nb_runner.run_all()
        assert "log = ['INIT', 'PROCESS', 'COMPLETE']" in nb_runner.get_output(2)
