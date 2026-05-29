"""Batch 260 – Itertools patterns with edits.

Tests itertools functions with data/function edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsEdits:
    """Itertools operation edit patterns."""



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
