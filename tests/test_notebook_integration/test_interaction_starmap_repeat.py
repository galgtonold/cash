"""
Interaction test: itertools starmap and repeat.
Tests starmap for unpacking arguments, repeat for infinite iterators,
islice for limiting, and cross-cell functional patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStarmapRepeat:
    """Test itertools starmap and repeat across cells."""

    def test_starmap_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: starmap with pairs
            "from itertools import starmap, repeat, islice\npairs = [(2, 3), (4, 5), (6, 7)]\nproducts = list(starmap(lambda x, y: x * y, pairs))\nprint(f'products={products}')",
            # Cell 2: starmap with pow
            "powers = list(starmap(pow, [(2, 10), (3, 5), (10, 3)]))\nprint(f'powers={powers}')",
            # Cell 3: repeat + islice
            "repeated = list(islice(repeat('hello', 5), 5))\nprint(f'repeated={repeated}')\nprint(f'count={len(repeated)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "products=[6, 20, 42]" in out1
        out2 = nb_runner.get_output(2)
        assert "powers=[1024, 243, 1000]" in out2
        out3 = nb_runner.get_output(3)
        assert "count=5" in out3

    def test_starmap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import starmap\ncoords = [(0, 0), (3, 4), (6, 8)]\nimport math\ndistances = list(starmap(math.hypot, coords))\nprint(f'distances={distances}')",
            "total_dist = sum(distances)\nprint(f'total={total_dist}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "distances=[0.0, 5.0, 10.0]" in nb_runner.get_output(1)
        assert "total=15.0" in nb_runner.get_output(2)

        # Edit coords
        nb_runner.set_cell_source(1, "from itertools import starmap\ncoords = [(0, 0), (3, 4), (5, 12)]\nimport math\ndistances = list(starmap(math.hypot, coords))\nprint(f'distances={distances}')")
        nb_runner.run_cells([1, 2])
        assert "distances=[0.0, 5.0, 13.0]" in nb_runner.get_output(1)
        assert "total=18.0" in nb_runner.get_output(2)

    def test_starmap_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import starmap\ndata = [(1, 'a'), (2, 'b'), (3, 'c')]\nformatted = list(starmap(lambda n, s: f'{n}:{s}', data))\nprint(f'formatted={formatted}')",
            "joined = ', '.join(formatted)\nprint(f'joined={joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted=['1:a', '2:b', '3:c']" in nb_runner.get_output(1)
        assert "joined=1:a, 2:b, 3:c" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "joined=1:a, 2:b, 3:c" in nb_runner.get_output(2)
