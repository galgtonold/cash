"""Batch 160 – Nested data structure interaction tests.

Tests where nested dicts, lists, and objects are modified
across cells and dependencies must propagate correctly.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestNestedDictEdits:
    """Edit cells producing/consuming nested dicts."""

    def test_edit_nested_key(self, nb_runner):
        """Change a nested dict key."""
        nb_runner.create_notebook([
            "config = {'db': {'host': 'localhost', 'port': 5432}}",
            "addr = f\"{config['db']['host']}:{config['db']['port']}\"\nprint(f'addr = {addr}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr = localhost:5432" in nb_runner.get_output(2)

        # Change the host
        nb_runner.set_cell_source(
            1, "config = {'db': {'host': '10.0.0.1', 'port': 5432}}"
        )
        nb_runner.run_all()
        assert "addr = 10.0.0.1:5432" in nb_runner.get_output(2)

    def test_add_nested_level(self, nb_runner):
        """Add a deeper nesting level."""
        nb_runner.create_notebook([
            "data = {'a': 1}",
            "total = sum(v if isinstance(v, int) else sum(v.values()) for v in data.values())\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1" in nb_runner.get_output(2)

        # Add nested dict
        nb_runner.set_cell_source(1, "data = {'a': 1, 'b': {'x': 10, 'y': 20}}")
        nb_runner.run_all()
        assert "total = 31" in nb_runner.get_output(2)

    def test_edit_dict_comprehension_source(self, nb_runner):
        """Edit dict comprehension inputs."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']\nvals = [1, 2, 3]",
            "mapping = dict(zip(keys, vals))\nprint(f'mapping = {mapping}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "keys = ['x', 'y', 'z']\nvals = [10, 20, 30]")
        nb_runner.run_all()
        assert "'x': 10" in nb_runner.get_output(2)


class TestNestedListEdits:
    """Edit cells producing/consuming nested lists."""

    def test_edit_2d_list(self, nb_runner):
        """Edit a 2D list source."""
        nb_runner.create_notebook([
            "matrix = [[1, 2], [3, 4]]",
            "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40], [50, 60]]")
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40, 50, 60]" in nb_runner.get_output(2)

    def test_nested_list_processing_chain(self, nb_runner):
        """Chain of nested list operations."""
        nb_runner.create_notebook([
            "raw = [[1, 2, 3], [4, 5, 6]]",
            "sums = [sum(row) for row in raw]",
            "total = sum(sums)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "raw = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "total = 100" in nb_runner.get_output(3)
