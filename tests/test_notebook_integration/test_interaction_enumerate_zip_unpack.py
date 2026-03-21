"""Batch 372: enumerate with start, zip with strict, and unpacking."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumerateZipUnpack:
    def test_enumerate_start(self, nb_runner):
        nb_runner.create_notebook([
            "items = ['apple', 'banana', 'cherry']",
            "numbered = list(enumerate(items, start=1))\nprint(f'numbered={numbered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "numbered=[(1, 'apple'), (2, 'banana'), (3, 'cherry')]" in nb_runner.get_output(2)

    def test_enumerate_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = ['x', 'y', 'z']",
            "pairs = {i: v for i, v in enumerate(data)}\nprint(f'pairs={pairs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pairs={0: 'x', 1: 'y', 2: 'z'}" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "data = ['a', 'b']")
        nb_runner.run_all()
        assert "pairs={0: 'a', 1: 'b'}" in nb_runner.get_output(2)

    def test_zip_unpack_pattern(self, nb_runner):
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Charlie']\nages = [30, 25, 35]\ncities = ['NY', 'LA', 'SF']",
            "records = list(zip(names, ages, cities))\noldest = max(records, key=lambda r: r[1])\nprint(f'oldest={oldest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "oldest=('Charlie', 35, 'SF')" in nb_runner.get_output(2)
