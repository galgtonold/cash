"""
Batch 289: Protocol / structural subtyping interaction tests.
Tests that editing classes implementing protocols properly invalidates
downstream cells that use protocol-based operations.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestProtocolInteraction:
    """Test protocol/structural subtyping patterns with cache invalidation."""

    def test_duck_typing_edit(self, nb_runner):
        """Editing a class used via duck typing should propagate."""
        nb_runner.create_notebook([
            (
                "class Dog:\n"
                "    def speak(self):\n"
                "        return 'Woof'\n"
                "class Cat:\n"
                "    def speak(self):\n"
                "        return 'Meow'"
            ),
            "animals = [Dog(), Cat()]",
            "sounds = [a.speak() for a in animals]",
            "result = ', '.join(sounds)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=Woof, Meow" in out

        # Edit class definitions
        nb_runner.set_cell_source(1, (
            "class Dog:\n"
            "    def speak(self):\n"
            "        return 'BARK'\n"
            "class Cat:\n"
            "    def speak(self):\n"
            "        return 'HISS'"
        ))
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=BARK, HISS" in out

    def test_callable_protocol_edit(self, nb_runner):
        """Editing callable objects used as strategies should propagate."""
        nb_runner.create_notebook([
            (
                "class Adder:\n"
                "    def __init__(self, n):\n"
                "        self.n = n\n"
                "    def __call__(self, x):\n"
                "        return x + self.n"
            ),
            "op = Adder(10)",
            "result = op(5)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=15" in out

        nb_runner.set_cell_source(2, "op = Adder(100)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=105" in out

    def test_iterable_protocol_edit(self, nb_runner):
        """Editing a custom iterable class should propagate."""
        nb_runner.create_notebook([
            (
                "class Range2:\n"
                "    def __init__(self, start, stop):\n"
                "        self.start = start\n"
                "        self.stop = stop\n"
                "    def __iter__(self):\n"
                "        current = self.start\n"
                "        while current < self.stop:\n"
                "            yield current\n"
                "            current += 2"
            ),
            "r = Range2(0, 10)",
            "vals = list(r)",
            "result = ','.join(str(v) for v in vals)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=0,2,4,6,8" in out

        nb_runner.set_cell_source(2, "r = Range2(1, 12)")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=1,3,5,7,9,11" in out
