"""
Batch 287: Descriptor protocol interaction tests.
Tests that editing descriptor-based attribute access logic
properly invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDescriptorInteraction:
    """Test descriptor protocol patterns with cache invalidation."""

    def test_property_descriptor_edit(self, nb_runner):
        """Editing a class with property descriptors should propagate."""
        nb_runner.create_notebook([
            (
                "class Circle:\n"
                "    def __init__(self, radius):\n"
                "        self._radius = radius\n"
                "    @property\n"
                "    def area(self):\n"
                "        return 3.14159 * self._radius ** 2"
            ),
            "c = Circle(5)",
            "a = round(c.area, 2)",
            "print(f'area={a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "area=78.54" in out

        nb_runner.set_cell_source(2, "c = Circle(10)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "area=314.16" in out

    def test_custom_descriptor_edit(self, nb_runner):
        """Editing a custom descriptor class should propagate."""
        nb_runner.create_notebook([
            (
                "class Validator:\n"
                "    def __init__(self, min_val, max_val):\n"
                "        self.min_val = min_val\n"
                "        self.max_val = max_val\n"
                "    def __set_name__(self, owner, name):\n"
                "        self.name = '_' + name\n"
                "    def __get__(self, obj, objtype=None):\n"
                "        return getattr(obj, self.name, None)\n"
                "    def __set__(self, obj, value):\n"
                "        if not (self.min_val <= value <= self.max_val):\n"
                "            raise ValueError(f'{value} not in [{self.min_val},{self.max_val}]')\n"
                "        setattr(obj, self.name, value)"
            ),
            (
                "class Sensor:\n"
                "    temperature = Validator(-50, 150)\n"
                "    def __init__(self, temp):\n"
                "        self.temperature = temp"
            ),
            "s = Sensor(25)\nval = s.temperature",
            "print(f'temp={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "temp=25" in out

        nb_runner.set_cell_source(3, "s = Sensor(99)\nval = s.temperature")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "temp=99" in out

    def test_cached_property_edit(self, nb_runner):
        """Editing data used by a cached_property should propagate."""
        nb_runner.create_notebook([
            "from functools import cached_property",
            (
                "class Stats:\n"
                "    def __init__(self, data):\n"
                "        self.data = data\n"
                "    @cached_property\n"
                "    def mean(self):\n"
                "        return sum(self.data) / len(self.data)"
            ),
            "st = Stats([10, 20, 30])\nm = st.mean",
            "print(f'mean={m}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mean=20.0" in out

        nb_runner.set_cell_source(3, "st = Stats([100, 200, 300])\nm = st.mean")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mean=200.0" in out
