"""Batch 80: State machine patterns — cash caching with FSM implementations."""
import textwrap
import pytest


@pytest.mark.stress
class TestStateMachine:
    """Test state machine patterns across cells."""

    def test_basic_state_machine(self, nb_runner):
        """Simple FSM across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class StateMachine:
                    def __init__(self):
                        self.state = 'idle'
                        self.transitions = {
                            ('idle', 'start'): 'running',
                            ('running', 'pause'): 'paused',
                            ('paused', 'resume'): 'running',
                            ('running', 'stop'): 'idle',
                            ('paused', 'stop'): 'idle',
                        }
                        self.history = ['idle']

                    def trigger(self, event):
                        key = (self.state, event)
                        if key in self.transitions:
                            self.state = self.transitions[key]
                            self.history.append(self.state)
                            return True
                        return False

                sm = StateMachine()
                for event in ['start', 'pause', 'resume', 'stop']:
                    sm.trigger(event)
                print(f"final={sm.state}")
                print(f"history={sm.history}")
            """),
            textwrap.dedent("""\
                unique_states = set(sm.history)
                print(f"visited={sorted(unique_states)}")
                print(f"transitions={len(sm.history) - 1}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final=idle" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "transitions=4" in out2

    def test_order_state_machine(self, nb_runner):
        """Order processing FSM across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Order:
                    VALID_TRANSITIONS = {
                        'pending': ['confirmed', 'cancelled'],
                        'confirmed': ['shipped', 'cancelled'],
                        'shipped': ['delivered'],
                        'delivered': [],
                        'cancelled': [],
                    }

                    def __init__(self, order_id):
                        self.order_id = order_id
                        self.status = 'pending'
                        self.log = []

                    def transition(self, new_status):
                        if new_status in self.VALID_TRANSITIONS.get(self.status, []):
                            old = self.status
                            self.status = new_status
                            self.log.append(f"{old}->{new_status}")
                            return True
                        return False

                o1 = Order('ORD-001')
                o1.transition('confirmed')
                o1.transition('shipped')
                o1.transition('delivered')

                o2 = Order('ORD-002')
                o2.transition('cancelled')
            """),
            textwrap.dedent("""\
                print(f"o1: {o1.status}, log={o1.log}")
                print(f"o2: {o2.status}, log={o2.log}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "o1: delivered" in out
        assert "o2: cancelled" in out

    def test_fsm_propagation(self, nb_runner):
        """State machine propagates when events change."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final=green" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            states = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
            current = 'green'
            steps = 6
            history = [current]
            for _ in range(steps):
                current = states[current]
                history.append(current)
        """))
        nb_runner.run_cells([1, 2])
        # 6 transitions: green→yellow→red→green→yellow→red→green, final=green
        assert "final=green" in nb_runner.get_output(2)
        assert len(nb_runner.get_output(2).split("history=")[1].split("]")[0].split(",")) == 7


@pytest.mark.stress
class TestEventDriven:
    """Test event-driven patterns."""

    def test_event_bus(self, nb_runner):
        """Simple event bus across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class EventBus:
                    def __init__(self):
                        self._handlers = {}
                        self._log = []

                    def on(self, event, handler):
                        self._handlers.setdefault(event, []).append(handler)

                    def emit(self, event, data=None):
                        for handler in self._handlers.get(event, []):
                            result = handler(data)
                            self._log.append((event, result))

                bus = EventBus()
                bus.on('data', lambda x: x * 2)
                bus.on('data', lambda x: x + 10)
                bus.emit('data', 5)
                bus.emit('data', 100)
                print(f"log_count={len(bus._log)}")
            """),
            textwrap.dedent("""\
                results = [r for _, r in bus._log]
                print(f"results={results}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log_count=4" in nb_runner.get_output(1)
        assert "results=[10, 15, 200, 110]" in nb_runner.get_output(2)
