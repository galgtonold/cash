"""
Batch 313: Sorting with custom key functions interaction tests.
Tests that editing sort keys or comparison functions properly
invalidates sorted outputs downstream.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestSortingKeyInteraction:
    """Test sorting with custom keys and cache invalidation."""

    def test_sort_by_key_edit(self, nb_runner):
        """Editing sort key function should propagate."""
        nb_runner.create_notebook([
            "items = [('banana', 3), ('apple', 1), ('cherry', 2)]",
            "def sort_key(item):\n    return item[0]",
            "sorted_items = sorted(items, key=sort_key)",
            "result = [x[0] for x in sorted_items]",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=['apple', 'banana', 'cherry']" in out

        # Change to sort by number (index 1)
        nb_runner.set_cell_source(2, "def sort_key(item):\n    return item[1]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=['apple', 'cherry', 'banana']" in out

    def test_sort_reverse_edit(self, nb_runner):
        """Editing sort direction should propagate."""
        nb_runner.create_notebook([
            "data = [5, 3, 8, 1, 9]",
            "ascending = True",
            "ordered = sorted(data, reverse=not ascending)",
            "print(f'ordered={ordered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "ordered=[1, 3, 5, 8, 9]" in out

        nb_runner.set_cell_source(2, "ascending = False")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "ordered=[9, 8, 5, 3, 1]" in out

    def test_multi_key_sort_edit(self, nb_runner):
        """Editing multi-key sort criteria should propagate."""
        nb_runner.create_notebook([
            "records = [('Alice', 85), ('Bob', 92), ('Charlie', 85), ('David', 92)]",
            "def multi_key(r):\n    return (-r[1], r[0])",
            "ranked = sorted(records, key=multi_key)",
            "names = [r[0] for r in ranked]",
            "print(f'names={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        # Sort by score desc then name asc: Bob(92), David(92), Alice(85), Charlie(85)
        assert "names=['Bob', 'David', 'Alice', 'Charlie']" in out

        # Change to sort by name only
        nb_runner.set_cell_source(2, "def multi_key(r):\n    return r[0]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "names=['Alice', 'Bob', 'Charlie', 'David']" in out
