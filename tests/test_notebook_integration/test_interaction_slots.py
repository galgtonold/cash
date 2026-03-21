"""
Batch 288: Slots and __slots__ interaction tests.
Tests that editing classes with __slots__ and their instances
properly invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestSlotsInteraction:
    """Test __slots__ class patterns with cache invalidation."""

    def test_slots_class_instance_edit(self, nb_runner):
        """Editing a slots-based class instance should propagate."""
        nb_runner.create_notebook([
            (
                "class Vector:\n"
                "    __slots__ = ('x', 'y')\n"
                "    def __init__(self, x, y):\n"
                "        self.x = x\n"
                "        self.y = y\n"
                "    def magnitude(self):\n"
                "        return (self.x**2 + self.y**2) ** 0.5"
            ),
            "v = Vector(3, 4)",
            "mag = round(v.magnitude(), 2)",
            "print(f'mag={mag}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mag=5.0" in out

        nb_runner.set_cell_source(2, "v = Vector(5, 12)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mag=13.0" in out

    def test_slots_class_definition_edit(self, nb_runner):
        """Editing the class definition with __slots__ should propagate."""
        nb_runner.create_notebook([
            (
                "class Record:\n"
                "    __slots__ = ('name', 'value')\n"
                "    def __init__(self, name, value):\n"
                "        self.name = name\n"
                "        self.value = value\n"
                "    def display(self):\n"
                "        return f'{self.name}={self.value}'"
            ),
            "r = Record('temp', 25)",
            "text = r.display()",
            "print(f'text={text}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "text=temp=25" in out

        # Edit class to change display format
        nb_runner.set_cell_source(1, (
            "class Record:\n"
            "    __slots__ = ('name', 'value')\n"
            "    def __init__(self, name, value):\n"
            "        self.name = name\n"
            "        self.value = value\n"
            "    def display(self):\n"
            "        return f'{self.name}: {self.value}'"
        ))
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "text=temp: 25" in out
