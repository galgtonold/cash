"""Batch 58: Design patterns — Observer, Strategy, Builder, State with cash caching."""
import textwrap
import pytest


@pytest.mark.stress
class TestObserverPattern:
    """Test Observer/Event pattern."""

    def test_observer_basic(self, nb_runner):
        """Observer pattern with publish/subscribe."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class EventEmitter:
                    def __init__(self):
                        self._listeners = {}
                        self._history = []

                    def on(self, event, callback):
                        self._listeners.setdefault(event, []).append(callback)

                    def emit(self, event, *args):
                        self._history.append((event, args))
                        for cb in self._listeners.get(event, []):
                            cb(*args)

                emitter = EventEmitter()
                results = []
                emitter.on('data', lambda x: results.append(f"got:{x}"))
                emitter.on('data', lambda x: results.append(f"also:{x}"))
            """),
            textwrap.dedent("""\
                emitter.emit('data', 42)
                emitter.emit('data', 99)
                print(f"results={results}")
                print(f"history_count={len(emitter._history)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "got:42" in out
        assert "also:99" in out
        assert "history_count=2" in out


@pytest.mark.stress
class TestStrategyPattern:
    """Test Strategy pattern."""

    def test_strategy_swap(self, nb_runner):
        """Strategy pattern with swappable algorithms."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Sorter:
                    def __init__(self, strategy=None):
                        self.strategy = strategy or sorted

                    def sort(self, data):
                        return self.strategy(data)

                def reverse_sort(data):
                    return sorted(data, reverse=True)

                def abs_sort(data):
                    return sorted(data, key=abs)

                default_sorter = Sorter()
                reverse_sorter = Sorter(reverse_sort)
                abs_sorter = Sorter(abs_sort)
            """),
            textwrap.dedent("""\
                data = [3, -1, 4, -1, 5, -9, 2, 6]
                r1 = default_sorter.sort(data)
                r2 = reverse_sorter.sort(data)
                r3 = abs_sorter.sort(data)
                print(f"default={r1}")
                print(f"reverse={r2}")
                print(f"abs={r3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "default=[-9, -1, -1, 2, 3, 4, 5, 6]" in out
        assert "reverse=[6, 5, 4, 3, 2, -1, -1, -9]" in out

    def test_strategy_change_propagates(self, nb_runner):
        """Changing strategy function propagates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def formatter(x):
                    return f"${x:.2f}"
            """),
            textwrap.dedent("""\
                prices = [10, 20.5, 3.99]
                formatted = [formatter(p) for p in prices]
                print(f"formatted={formatted}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "$10.00" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def formatter(x):
                return f"EUR {x:.2f}"
        """))
        nb_runner.run_all()
        assert "EUR 10.00" in nb_runner.get_output(2)


@pytest.mark.stress
class TestBuilderPattern:
    """Test Builder pattern."""

    def test_query_builder(self, nb_runner):
        """Builder pattern for SQL-like query construction."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class QueryBuilder:
                    def __init__(self, table):
                        self._table = table
                        self._conditions = []
                        self._order = None
                        self._limit = None

                    def where(self, condition):
                        self._conditions.append(condition)
                        return self

                    def order_by(self, field):
                        self._order = field
                        return self

                    def limit(self, n):
                        self._limit = n
                        return self

                    def build(self):
                        sql = f"SELECT * FROM {self._table}"
                        if self._conditions:
                            sql += " WHERE " + " AND ".join(self._conditions)
                        if self._order:
                            sql += f" ORDER BY {self._order}"
                        if self._limit:
                            sql += f" LIMIT {self._limit}"
                        return sql
            """),
            textwrap.dedent("""\
                query = (QueryBuilder("users")
                    .where("age > 18")
                    .where("active = true")
                    .order_by("name")
                    .limit(10)
                    .build())
                print(f"query={query}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "SELECT * FROM users" in out
        assert "age > 18" in out
        assert "LIMIT 10" in out


@pytest.mark.stress
class TestStateMachinePattern:
    """Test State Machine pattern."""

    def test_simple_state_machine(self, nb_runner):
        """State machine with transitions."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class StateMachine:
                    def __init__(self, initial):
                        self.state = initial
                        self.transitions = {}
                        self.history = [initial]

                    def add_transition(self, from_state, event, to_state):
                        self.transitions[(from_state, event)] = to_state

                    def trigger(self, event):
                        key = (self.state, event)
                        if key in self.transitions:
                            self.state = self.transitions[key]
                            self.history.append(self.state)
                            return True
                        return False

                sm = StateMachine('idle')
                sm.add_transition('idle', 'start', 'running')
                sm.add_transition('running', 'pause', 'paused')
                sm.add_transition('paused', 'resume', 'running')
                sm.add_transition('running', 'stop', 'idle')
            """),
            textwrap.dedent("""\
                sm.trigger('start')
                sm.trigger('pause')
                sm.trigger('resume')
                sm.trigger('stop')
                print(f"state={sm.state}")
                print(f"history={sm.history}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "state=idle" in out
        assert "history=['idle', 'running', 'paused', 'running', 'idle']" in out

    def test_state_machine_change(self, nb_runner):
        """Changing state machine transitions propagates."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "states=['B', 'C', 'A', 'B', 'C', 'A']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class FSM:
                def __init__(self):
                    self.state = 'X'
                    self.rules = {'X': 'Y', 'Y': 'X'}
                def step(self):
                    self.state = self.rules.get(self.state, self.state)
                    return self.state

            fsm = FSM()
        """))
        nb_runner.run_all()
        assert "states=['Y', 'X', 'Y', 'X', 'Y', 'X']" in nb_runner.get_output(2)
