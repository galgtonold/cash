"""State machine patterns — cash caching with FSM implementations."""

import textwrap

import pytest


@pytest.mark.stress
class TestStateMachine:
    """Test state machine patterns across cells."""

    def test_fsm_propagation(self, nb_runner):
        """State machine propagates when events change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                states = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
                current = 'green'
                steps = 3
                history = [current]
                for _ in range(steps):
                    current = states[current]
                    history.append(current)
            """),
                textwrap.dedent("""\
                print(f"history={history}")
                print(f"final={history[-1]}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final=green" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            states = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
            current = 'green'
            steps = 6
            history = [current]
            for _ in range(steps):
                current = states[current]
                history.append(current)
        """),
        )
        nb_runner.run_cells([1, 2])
        # 6 transitions: green→yellow→red→green→yellow→red→green, final=green
        assert "final=green" in nb_runner.get_output(2)
        assert len(nb_runner.get_output(2).split("history=")[1].split("]")[0].split(",")) == 7
