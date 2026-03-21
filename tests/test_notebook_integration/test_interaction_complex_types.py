"""Batch 118 – Complex data types + cell edit interaction tests.

Tests that exercise cache behavior with complex data types:
dicts, lists, nested structures, sets, tuples, etc.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestDictEdits:
    """Dict values with cell edits."""

    def test_dict_creation_edit(self, nb_runner):
        """Edit a dict creation cell."""
        nb_runner.create_notebook([
            "config = {'a': 1, 'b': 2}",
            "total = sum(config.values())\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "config = {'a': 10, 'b': 20, 'c': 30}")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_dict_access_edit(self, nb_runner):
        """Edit how a dict is accessed."""
        nb_runner.create_notebook([
            "data = {'x': 10, 'y': 20, 'z': 30}",
            "val = data['x']\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "val = data['z']\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(2)

    def test_nested_dict_edit(self, nb_runner):
        """Edit a nested dict."""
        nb_runner.create_notebook([
            "data = {'outer': {'inner': 42}}",
            "val = data['outer']['inner']\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = {'outer': {'inner': 99}}")
        nb_runner.run_all()
        assert "val = 99" in nb_runner.get_output(2)


class TestListEdits:
    """List operations with cell edits."""

    def test_list_slice_edit(self, nb_runner):
        """Edit a list and downstream slice operation."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "subset = data[:3]\nprint(f'subset = {subset}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "subset = [1, 2, 3]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "subset = [10, 20, 30]" in nb_runner.get_output(2)

    def test_list_operation_edit(self, nb_runner):
        """Edit the list operation."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "result = sum(data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = max(data)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(2)


class TestTupleSetEdits:
    """Tuples and sets with edits."""

    def test_tuple_unpack_edit(self, nb_runner):
        """Edit a tuple unpacking cell."""
        nb_runner.create_notebook([
            "pair = (10, 20)",
            "a, b = pair\nprint(f'a = {a}, b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10, b = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "pair = (100, 200)")
        nb_runner.run_all()
        assert "a = 100, b = 200" in nb_runner.get_output(2)

    def test_set_operations_edit(self, nb_runner):
        """Edit set operations."""
        nb_runner.create_notebook([
            "s1 = {1, 2, 3}\ns2 = {2, 3, 4}",
            "result = s1 & s2\nprint(f'result = {sorted(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3]" in nb_runner.get_output(2)

        # Change to union
        nb_runner.set_cell_source(2, "result = s1 | s2\nprint(f'result = {sorted(result)}')")
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4]" in nb_runner.get_output(2)


class TestStringOperations:
    """String manipulation with edits."""

    def test_string_format_edit(self, nb_runner):
        """Edit string formatting."""
        nb_runner.create_notebook([
            "name = 'World'",
            "greeting = f'Hello, {name}!'\nprint(greeting)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello, World!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "name = 'Cash'")
        nb_runner.run_all()
        assert "Hello, Cash!" in nb_runner.get_output(2)

    def test_string_join_edit(self, nb_runner):
        """Edit string join operations."""
        nb_runner.create_notebook([
            "words = ['Hello', 'World']",
            "sentence = ' '.join(words)\nprint(sentence)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "words = ['Foo', 'Bar', 'Baz']")
        nb_runner.run_all()
        assert "Foo Bar Baz" in nb_runner.get_output(2)


class TestComplexDataFlowEdits:
    """Complex data flowing through multiple cells with edits."""

    def test_dict_to_list_to_sum(self, nb_runner):
        """Dict → list extraction → sum, edit the dict."""
        nb_runner.create_notebook([
            "scores = {'math': 90, 'english': 85, 'science': 95}",
            "values = list(scores.values())",
            "total = sum(values)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 270" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1, "scores = {'math': 100, 'english': 100, 'science': 100}"
        )
        nb_runner.run_all()
        assert "total = 300" in nb_runner.get_output(3)

    def test_list_filter_transform_aggregate(self, nb_runner):
        """List → filter → transform → aggregate, edit filter."""
        nb_runner.create_notebook([
            "data = list(range(10))",
            "filtered = [x for x in data if x > 5]",
            "transformed = [x * 10 for x in filtered]",
            "result = sum(transformed)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(4)

        # Change filter condition
        nb_runner.set_cell_source(2, "filtered = [x for x in data if x > 2]")
        nb_runner.run_all()
        assert "result = 420" in nb_runner.get_output(4)
