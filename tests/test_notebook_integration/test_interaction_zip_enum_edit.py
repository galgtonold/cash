"""Batch 277 – Zip and enumerate patterns with edits.

Tests zip, enumerate, and parallel iteration with data edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipEnumEdits:
    """Zip/enumerate edit patterns."""

    def test_zip_edit_one_list(self, nb_runner):
        """Edit one of two zipped lists."""
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Charlie']",
            "scores = [90, 85, 78]",
            "pairs = list(zip(names, scores))\nprint(f'pairs = {pairs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('Alice', 90)" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "scores = [100, 95, 88]")
        nb_runner.run_all()
        assert "('Alice', 100)" in nb_runner.get_output(3)
        assert "('Charlie', 88)" in nb_runner.get_output(3)

    def test_enumerate_with_edit(self, nb_runner):
        """Edit list, enumerate indexes correctly reflect."""
        nb_runner.create_notebook([
            "items = ['apple', 'banana', 'cherry']",
            "indexed = list(enumerate(items, start=1))\nprint(f'indexed = {indexed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(1, 'apple')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "items = ['x', 'y']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(1, 'x')" in out
        assert "(2, 'y')" in out

    def test_zip_longest_edit(self, nb_runner):
        """Edit data in zip_longest scenario."""
        nb_runner.create_notebook([
            "from itertools import zip_longest\na = [1, 2, 3]\nb = ['x', 'y']",
            "result = list(zip_longest(a, b, fillvalue='?'))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(3, '?')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "from itertools import zip_longest\na = [1]\nb = ['x', 'y', 'z']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(1, 'x')" in out
        assert "('?', 'z')" in out
