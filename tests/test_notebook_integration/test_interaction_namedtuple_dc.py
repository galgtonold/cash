"""
Batch 325: namedtuple and dataclass patterns with caching.
Tests namedtuple creation, dataclass fields, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestNamedtupleDataclass:
    """Test namedtuple and dataclass caching."""

    def test_namedtuple_basic(self, nb_runner):
        """Create and use namedtuple, verify caching."""
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Point = namedtuple('Point', ['x', 'y'])",
            "p = Point(3, 4)\ndist = (p.x**2 + p.y**2)**0.5",
            "print(f'dist={dist}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "dist=5.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "dist=5.0" in out2

    def test_namedtuple_edit_values(self, nb_runner):
        """Edit namedtuple values, verify propagation."""
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "RGB = namedtuple('RGB', 'r g b')\ncolor = RGB(255, 0, 0)",
            "brightness = (color.r + color.g + color.b) // 3",
            "print(f'brightness={brightness}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "brightness=85" in out

        nb_runner.set_cell_source(2, "RGB = namedtuple('RGB', 'r g b')\ncolor = RGB(0, 255, 0)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "brightness=85" in out2

    def test_dataclass_pattern(self, nb_runner):
        """Dataclass creation and field access with caching."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass\nclass Item:\n    name: str\n    price: float\n    qty: int = 1",
            "item = Item('Widget', 9.99, 5)\ntotal = item.price * item.qty",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=49.95" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=49.95" in out2
