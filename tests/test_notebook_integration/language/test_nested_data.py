"""Deeply nested lists and dicts built and edited across cells."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.stress
class TestNestedContainers:
    """Test deeply nested containers across cells."""

    def test_nested_dict_access(self, nb_runner):
        """Deeply nested dict accessed across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                config = {
                    'database': {
                        'host': 'localhost',
                        'port': 5432,
                        'credentials': {
                            'user': 'admin',
                            'password': 'secret'
                        }
                    },
                    'cache': {
                        'ttl': 300
                    }
                }
            """),
                textwrap.dedent("""\
                host = config['database']['host']
                user = config['database']['credentials']['user']
                ttl = config['cache']['ttl']
                print(f"host={host} user={user} ttl={ttl}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost user=admin ttl=300" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNestedListEdits:
    """Edit cells producing/consuming nested lists."""

    def test_nested_list_processing_chain(self, nb_runner):
        """Chain of nested list operations."""
        nb_runner.create_notebook(
            [
                "raw = [[1, 2, 3], [4, 5, 6]]",
                "sums = [sum(row) for row in raw]",
                "total = sum(sums)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "raw = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "total = 100" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNestedDictEdits:
    """Edit cells producing/consuming nested dicts."""

    def test_edit_nested_key(self, nb_runner):
        """Change a nested dict key."""
        nb_runner.create_notebook(
            [
                "config = {'db': {'host': 'localhost', 'port': 5432}}",
                "addr = f\"{config['db']['host']}:{config['db']['port']}\"\nprint(f'addr = {addr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr = localhost:5432" in nb_runner.get_output(2)

        # Change the host
        nb_runner.set_cell_source(1, "config = {'db': {'host': '10.0.0.1', 'port': 5432}}")
        nb_runner.run_all()
        assert "addr = 10.0.0.1:5432" in nb_runner.get_output(2)

    def test_add_nested_level(self, nb_runner):
        """Add a deeper nesting level."""
        nb_runner.create_notebook(
            [
                "data = {'a': 1}",
                "total = sum(v if isinstance(v, int) else sum(v.values()) for v in data.values())\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1" in nb_runner.get_output(2)

        # Add nested dict
        nb_runner.set_cell_source(1, "data = {'a': 1, 'b': {'x': 10, 'y': 20}}")
        nb_runner.run_all()
        assert "total = 31" in nb_runner.get_output(2)

    def test_edit_dict_comprehension_source(self, nb_runner):
        """Edit dict comprehension inputs."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']\nvals = [1, 2, 3]",
                "mapping = dict(zip(keys, vals))\nprint(f'mapping = {mapping}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "keys = ['x', 'y', 'z']\nvals = [10, 20, 30]")
        nb_runner.run_all()
        assert "'x': 10" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDeepNestingEdits:
    """Editing deeply nested and complex data structures."""

    def test_edit_nested_dict_value(self, nb_runner):
        """Edit a nested dict value at depth 2 and verify propagation."""
        nb_runner.create_notebook(
            [
                "config = {'db': {'host': 'localhost', 'port': 5432}, 'debug': True}",
                "host = config['db']['host']\nport = config['db']['port']\nprint(f'host={host} port={port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=5432" in nb_runner.get_output(2)

        # Change host
        nb_runner.set_cell_source(1, "config = {'db': {'host': '10.0.0.1', 'port': 5432}, 'debug': True}")
        nb_runner.run_all()
        assert "host=10.0.0.1 port=5432" in nb_runner.get_output(2)

    def test_edit_list_of_records(self, nb_runner):
        """Edit a list of dicts (records pattern)."""
        nb_runner.create_notebook(
            [
                "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}]",
                "names = [r['name'] for r in records]\nprint(f'names = {names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(2)

        # Add a record
        nb_runner.set_cell_source(
            1,
            "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}, {'name': 'Charlie', 'score': 95}]",
        )
        nb_runner.run_all()
        assert "Charlie" in nb_runner.get_output(2)

    def test_edit_dict_with_tuple_keys(self, nb_runner):
        """Edit a dict with tuple keys."""
        nb_runner.create_notebook(
            [
                "grid = {(0, 0): 'X', (0, 1): 'O', (1, 0): '.'}",
                "val = grid.get((0, 0), '.')\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = X" in nb_runner.get_output(2)

        # Change the value
        nb_runner.set_cell_source(1, "grid = {(0, 0): 'O', (0, 1): 'O', (1, 0): '.'}")
        nb_runner.run_all()
        assert "val = O" in nb_runner.get_output(2)

    def test_edit_3_level_deep_nested(self, nb_runner):
        """Edit a deeply nested structure (3+ levels)."""
        nb_runner.create_notebook(
            [
                "tree = {'a': {'b': {'c': 42}}}",
                "leaf = tree['a']['b']['c']\nprint(f'leaf = {leaf}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "leaf = 42" in nb_runner.get_output(2)

        # Change deep value
        nb_runner.set_cell_source(1, "tree = {'a': {'b': {'c': 99}}}")
        nb_runner.run_all()
        assert "leaf = 99" in nb_runner.get_output(2)
