"""complex class interactions across multiple cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestCrossCellClassInteractions:
    """Classes defined in one cell, used in another."""

    def test_observer_pattern_cross_cell(self, nb_runner):
        """Observer pattern across cells with change propagation."""
        nb_runner.create_notebook(
            [
                "event_name = 'click'",
                textwrap.dedent("""\
                class EventBus:
                    def __init__(self):
                        self.listeners = {}
                        self.log = []
                    def on(self, event, fn):
                        self.listeners.setdefault(event, []).append(fn)
                    def emit(self, event, data=None):
                        for fn in self.listeners.get(event, []):
                            result = fn(data)
                            self.log.append(result)

                bus = EventBus()
                bus.on(event_name, lambda d: f"handler1: {d}")
                bus.on(event_name, lambda d: f"handler2: {d}")
                bus.emit(event_name, "test_data")
                logged = bus.log[:]
            """),
                "print(f'logged={logged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "handler1: test_data" in out
        assert "handler2: test_data" in out

        nb_runner.set_cell_source(1, "event_name = 'submit'")
        nb_runner.run_cells([1, 2, 3])
        out2 = nb_runner.get_output(3)
        assert "handler1: test_data" in out2
