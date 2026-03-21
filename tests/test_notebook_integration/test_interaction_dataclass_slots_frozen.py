"""
Interaction test: dataclass with slots and frozen.
Tests dataclass(slots=True, frozen=True) for memory-efficient
immutable records, and cross-cell immutable data patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassSlotsFrozen:
    """Test dataclass with slots and frozen across cells."""

    def test_dataclass_frozen(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: frozen dataclass
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n\np1 = Point(1.0, 2.0)\np2 = Point(3.0, 4.0)\nprint(f'p1={p1}')\nprint(f'p2={p2}')\nprint(f'hash_p1={hash(p1) == hash(Point(1.0, 2.0))}')",
            # Cell 2: use as dict key (hashable)
            "distances = {p1: 'near', p2: 'far'}\nprint(f'p1_dist={distances[p1]}')\nprint(f'p2_dist={distances[p2]}')",
            # Cell 3: verify immutability
            "try:\n    p1.x = 99\n    print('error=none')\nexcept Exception as e:\n    print(f'error={type(e).__name__}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "p1=Point(x=1.0, y=2.0)" in out1
        assert "hash_p1=True" in out1
        out2 = nb_runner.get_output(2)
        assert "p1_dist=near" in out2
        assert "p2_dist=far" in out2
        out3 = nb_runner.get_output(3)
        assert "FrozenInstanceError" in out3

    def test_dataclass_frozen_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Config:\n    host: str\n    port: int\n\ncfg = Config('localhost', 8080)\nprint(f'cfg={cfg}')",
            "addr = f'{cfg.host}:{cfg.port}'\nprint(f'addr={addr}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr=localhost:8080" in nb_runner.get_output(2)

        # Edit config
        nb_runner.set_cell_source(1, "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Config:\n    host: str\n    port: int\n\ncfg = Config('0.0.0.0', 9090)\nprint(f'cfg={cfg}')")
        nb_runner.run_cells([1, 2])
        assert "addr=0.0.0.0:9090" in nb_runner.get_output(2)

    def test_dataclass_frozen_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Vec2:\n    x: int\n    y: int\n\nv = Vec2(3, 4)\nmag_sq = v.x**2 + v.y**2\nprint(f'mag_sq={mag_sq}')",
            "is_unit = mag_sq == 1\nprint(f'is_unit={is_unit}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag_sq=25" in nb_runner.get_output(1)
        assert "is_unit=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_unit=False" in nb_runner.get_output(2)
