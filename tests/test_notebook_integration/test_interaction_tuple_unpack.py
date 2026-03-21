"""Batch 178 – Tuple unpacking and multi-return interaction tests.

Tests editing functions that return tuples, and editing
unpacking patterns in downstream cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestTupleUnpackingEdits:
    """Editing tuple unpacking patterns."""

    def test_edit_multi_return_function(self, nb_runner):
        """Edit a function that returns a tuple."""
        nb_runner.create_notebook([
            "def stats(data):\n    return min(data), max(data)",
            "lo, hi = stats([3, 1, 4, 1, 5])\nprint(f'lo={lo} hi={hi}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=5" in nb_runner.get_output(2)

        # Change function to return mean too
        nb_runner.set_cell_source(
            1,
            "def stats(data):\n    return min(data), max(data), sum(data)/len(data)",
        )
        nb_runner.set_cell_source(
            2,
            "lo, hi, avg = stats([3, 1, 4, 1, 5])\nprint(f'lo={lo} hi={hi} avg={avg}')",
        )
        nb_runner.run_all()
        assert "lo=1 hi=5 avg=2.8" in nb_runner.get_output(2)

    def test_edit_unpacking_target(self, nb_runner):
        """Edit the variables that receive unpacked values."""
        nb_runner.create_notebook([
            "pair = (10, 20)  # source tuple",
            "a, b = pair\nprint(f'a={a} b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=20" in nb_runner.get_output(2)

        # Change source
        nb_runner.set_cell_source(1, "pair = (100, 200)  # source tuple bigger")
        nb_runner.run_all()
        assert "a=100 b=200" in nb_runner.get_output(2)

    def test_star_unpacking_edit(self, nb_runner):
        """Edit star unpacking patterns."""
        nb_runner.create_notebook([
            "items = [1, 2, 3, 4, 5]  # items to unpack",
            "first, *rest = items\nprint(f'first={first} rest={rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1 rest=[2, 3, 4, 5]" in nb_runner.get_output(2)

        # Change to different unpacking
        nb_runner.set_cell_source(
            2, "*start, last = items\nprint(f'start={start} last={last}')"
        )
        nb_runner.run_all()
        assert "start=[1, 2, 3, 4] last=5" in nb_runner.get_output(2)


class TestDictUnpackingEdits:
    """Dict unpacking patterns."""

    def test_edit_dict_values_method(self, nb_runner):
        """Edit dict and verify unpacking updates."""
        nb_runner.create_notebook([
            "config = {'host': 'localhost', 'port': 8080}",
            "host = config['host']\nport = config['port']\nprint(f'addr={host}:{port}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr=localhost:8080" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "config = {'host': '0.0.0.0', 'port': 9090}"
        )
        nb_runner.run_all()
        assert "addr=0.0.0.0:9090" in nb_runner.get_output(2)

    def test_nested_unpacking_edit(self, nb_runner):
        """Edit nested tuple unpacking."""
        nb_runner.create_notebook([
            "data = ((1, 2), (3, 4))  # nested tuples",
            "(a, b), (c, d) = data\ntotal = a + b + c + d\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = ((10, 20), (30, 40))  # nested tuples bigger")
        nb_runner.run_all()
        assert "total = 100" in nb_runner.get_output(2)
