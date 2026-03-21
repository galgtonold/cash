"""Batch 227 – Deep nesting and complex structure edit tests.

Tests editing cells with deeply nested data structures, mixed types,
and complex data patterns to verify proper cache invalidation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDeepNestingEdits:
    """Editing deeply nested and complex data structures."""

    def test_edit_nested_dict_value(self, nb_runner):
        """Edit a nested dict value at depth 2 and verify propagation."""
        nb_runner.create_notebook([
            "config = {'db': {'host': 'localhost', 'port': 5432}, 'debug': True}",
            "host = config['db']['host']\nport = config['db']['port']\nprint(f'host={host} port={port}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=5432" in nb_runner.get_output(2)

        # Change host
        nb_runner.set_cell_source(1, "config = {'db': {'host': '10.0.0.1', 'port': 5432}, 'debug': True}")
        nb_runner.run_all()
        assert "host=10.0.0.1 port=5432" in nb_runner.get_output(2)

    def test_edit_list_of_records(self, nb_runner):
        """Edit a list of dicts (records pattern)."""
        nb_runner.create_notebook([
            "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}]",
            "names = [r['name'] for r in records]\nprint(f'names = {names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(2)

        # Add a record
        nb_runner.set_cell_source(1, "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}, {'name': 'Charlie', 'score': 95}]")
        nb_runner.run_all()
        assert "Charlie" in nb_runner.get_output(2)

    def test_edit_dict_with_tuple_keys(self, nb_runner):
        """Edit a dict with tuple keys."""
        nb_runner.create_notebook([
            "grid = {(0, 0): 'X', (0, 1): 'O', (1, 0): '.'}",
            "val = grid.get((0, 0), '.')\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = X" in nb_runner.get_output(2)

        # Change the value
        nb_runner.set_cell_source(1, "grid = {(0, 0): 'O', (0, 1): 'O', (1, 0): '.'}")
        nb_runner.run_all()
        assert "val = O" in nb_runner.get_output(2)

    def test_edit_3_level_deep_nested(self, nb_runner):
        """Edit a deeply nested structure (3+ levels)."""
        nb_runner.create_notebook([
            "tree = {'a': {'b': {'c': 42}}}",
            "leaf = tree['a']['b']['c']\nprint(f'leaf = {leaf}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "leaf = 42" in nb_runner.get_output(2)

        # Change deep value
        nb_runner.set_cell_source(1, "tree = {'a': {'b': {'c': 99}}}")
        nb_runner.run_all()
        assert "leaf = 99" in nb_runner.get_output(2)
