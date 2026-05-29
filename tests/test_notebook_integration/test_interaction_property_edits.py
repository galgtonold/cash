"""Batch 181 – Property and descriptor pattern interaction tests.

Tests editing property definitions, getters/setters,
and descriptor protocols across cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestPropertyEdits:
    """Editing property definitions."""

    def test_edit_property_getter(self, nb_runner):
        """Edit a property getter."""
        nb_runner.create_notebook([
            "class Temp:\n    def __init__(self, c):\n        self._c = c\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32",
            "t = Temp(100)\nprint(f'f = {t.fahrenheit}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 212.0" in nb_runner.get_output(2)

        # Change to Kelvin
        nb_runner.set_cell_source(
            1,
            "class Temp:\n    def __init__(self, c):\n        self._c = c\n    @property\n    def fahrenheit(self):\n        return self._c + 273.15",
        )
        nb_runner.run_all()
        assert "f = 373.15" in nb_runner.get_output(2)

    def test_add_property_setter(self, nb_runner):
        """Add a property setter to an existing class."""
        nb_runner.create_notebook([
            "class Box:\n    def __init__(self, w, h):\n        self._w = w\n        self._h = h\n    @property\n    def area(self):\n        return self._w * self._h",
            "b = Box(3, 4)\nprint(f'area = {b.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 12" in nb_runner.get_output(2)

        # Add volume
        nb_runner.set_cell_source(
            1,
            "class Box:\n    def __init__(self, w, h, d=1):\n        self._w = w\n        self._h = h\n        self._d = d\n    @property\n    def area(self):\n        return self._w * self._h\n    @property\n    def volume(self):\n        return self._w * self._h * self._d",
        )
        nb_runner.set_cell_source(2, "b = Box(3, 4, 5)\nprint(f'area={b.area} vol={b.volume}')")
        nb_runner.run_all()
        assert "area=12 vol=60" in nb_runner.get_output(2)


class TestClassMethodEdits:
    """Editing class/static methods."""

    def test_edit_classmethod(self, nb_runner):
        """Edit a classmethod."""
        nb_runner.create_notebook([
            "class Counter:\n    _count = 0\n    @classmethod\n    def increment(cls):\n        cls._count += 1\n        return cls._count",
            "a = Counter.increment()\nb = Counter.increment()\nprint(f'a={a} b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=2" in nb_runner.get_output(2)

        # Change to increment by 10
        nb_runner.set_cell_source(
            1,
            "class Counter:\n    _count = 0\n    @classmethod\n    def increment(cls):\n        cls._count += 10\n        return cls._count",
        )
        nb_runner.run_all()
        assert "a=10 b=20" in nb_runner.get_output(2)

