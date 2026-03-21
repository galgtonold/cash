"""Batch 99 – complex class interactions across multiple cells."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestCrossCellClassInteractions:
    """Classes defined in one cell, used in another."""

    def test_class_composition_cross_cell(self, nb_runner):
        """Class defined in cell 1, composed in cell 2, used in cell 3."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Engine:
                    def __init__(self, hp):
                        self.hp = hp
                    def describe(self):
                        return f"{self.hp}hp"
            """),
            textwrap.dedent("""\
                class Car:
                    def __init__(self, model, engine):
                        self.model = model
                        self.engine = engine
                    def info(self):
                        return f"{self.model} ({self.engine.describe()})"
                car = Car("Sedan", Engine(200))
            """),
            "print(f'car={car.info()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "car=Sedan (200hp)" in nb_runner.get_output(3)

    def test_inheritance_cross_cell(self, nb_runner):
        """Base in cell 1, derived in cell 2, usage in cell 3."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Shape:
                    def area(self):
                        raise NotImplementedError
                    def describe(self):
                        return f"{type(self).__name__}: area={self.area():.2f}"
            """),
            textwrap.dedent("""\
                import math
                class Circle(Shape):
                    def __init__(self, r):
                        self.r = r
                    def area(self):
                        return math.pi * self.r ** 2

                class Rectangle(Shape):
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    def area(self):
                        return self.w * self.h

                shapes = [Circle(5), Rectangle(3, 4)]
                descriptions = [s.describe() for s in shapes]
            """),
            "for d in descriptions:\n    print(d)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Circle" in out
        assert "78.54" in out
        assert "Rectangle" in out
        assert "12.00" in out

    def test_strategy_pattern_cross_cell(self, nb_runner):
        """Strategy pattern across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Sorter:
                    def __init__(self, strategy):
                        self.strategy = strategy
                    def sort(self, data):
                        return self.strategy(data)
            """),
            textwrap.dedent("""\
                def ascending(data):
                    return sorted(data)
                def descending(data):
                    return sorted(data, reverse=True)
                def by_length(data):
                    return sorted(data, key=len)
            """),
            textwrap.dedent("""\
                words = ['banana', 'apple', 'cherry', 'date']
                r1 = Sorter(ascending).sort(words)
                r2 = Sorter(descending).sort(words)
                r3 = Sorter(by_length).sort(words)
            """),
            "print(f'asc={r1}')\nprint(f'desc={r2}')\nprint(f'len={r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "apple" in out
        assert "asc=" in out
        assert "desc=" in out
        assert "len=" in out

    def test_observer_pattern_cross_cell(self, nb_runner):
        """Observer pattern across cells with change propagation."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "handler1: test_data" in out
        assert "handler2: test_data" in out

        nb_runner.set_cell_source(1, "event_name = 'submit'")
        nb_runner.run_cells([1, 2, 3])
        out2 = nb_runner.get_output(3)
        assert "handler1: test_data" in out2
