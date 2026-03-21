"""Batch 260 – Itertools patterns with edits.

Tests itertools functions with data/function edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsEdits:
    """Itertools operation edit patterns."""

    def test_chain_edit(self, nb_runner):
        """Edit data in itertools.chain."""
        nb_runner.create_notebook([
            "from itertools import chain\na = [1, 2, 3]\nb = [4, 5, 6]",
            "result = list(chain(a, b))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "from itertools import chain\na = [10, 20]\nb = [30, 40, 50]")
        nb_runner.run_all()
        assert "result = [10, 20, 30, 40, 50]" in nb_runner.get_output(2)

    def test_product_edit(self, nb_runner):
        """Edit product inputs."""
        nb_runner.create_notebook([
            "from itertools import product\ncolors = ['R', 'G']\nsizes = ['S', 'L']",
            "combos = list(product(colors, sizes))\nprint(f'combos = {combos}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('R', 'S')" in nb_runner.get_output(2)
        assert "('G', 'L')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from itertools import product\ncolors = ['R', 'G', 'B']\nsizes = ['M']",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('R', 'M')" in out
        assert "('B', 'M')" in out

    def test_groupby_edit(self, nb_runner):
        """Edit data before groupby."""
        nb_runner.create_notebook([
            "from itertools import groupby\ndata = [('a', 1), ('a', 2), ('b', 3), ('b', 4)]",
            "groups = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}\nresult = {k: len(v) for k, v in groups.items()}\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 2" in out

        nb_runner.set_cell_source(
            1,
            "from itertools import groupby\ndata = [('x', 1), ('x', 2), ('x', 3), ('y', 4)]",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'x': 3" in out2
        assert "'y': 1" in out2
