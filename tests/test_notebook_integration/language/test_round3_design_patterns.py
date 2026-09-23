"""Design patterns — Observer, Strategy, Builder, State with cash caching."""

import textwrap

import pytest


@pytest.mark.stress
class TestStrategyPattern:
    """Test Strategy pattern."""

    def test_strategy_change_propagates(self, nb_runner):
        """Changing strategy function propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def formatter(x):
                    return f"${x:.2f}"
            """),
                textwrap.dedent("""\
                prices = [10, 20.5, 3.99]
                formatted = [formatter(p) for p in prices]
                print(f"formatted={formatted}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "$10.00" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def formatter(x):
                return f"EUR {x:.2f}"
        """),
        )
        nb_runner.run_all()
        assert "EUR 10.00" in nb_runner.get_output(2)


@pytest.mark.stress
class TestStateMachinePattern:
    """Test State Machine pattern."""

    def test_state_machine_change(self, nb_runner):
        """Changing state machine transitions propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class FSM:
                    def __init__(self):
                        self.state = 'A'
                        self.rules = {'A': 'B', 'B': 'C', 'C': 'A'}
                    def step(self):
                        self.state = self.rules.get(self.state, self.state)
                        return self.state

                fsm = FSM()
            """),
                textwrap.dedent("""\
                states = [fsm.step() for _ in range(6)]
                print(f"states={states}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "states=['B', 'C', 'A', 'B', 'C', 'A']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class FSM:
                def __init__(self):
                    self.state = 'X'
                    self.rules = {'X': 'Y', 'Y': 'X'}
                def step(self):
                    self.state = self.rules.get(self.state, self.state)
                    return self.state

            fsm = FSM()
        """),
        )
        nb_runner.run_all()
        assert "states=['Y', 'X', 'Y', 'X', 'Y', 'X']" in nb_runner.get_output(2)
