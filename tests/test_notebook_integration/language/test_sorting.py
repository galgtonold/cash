"""sorted() and sort keys across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


# Sorting and ordering pattern interaction tests.
#
# Tests editing sort keys, reverse flags, custom comparators,
# and sorted data propagation.
@pytest.mark.upstream
class TestSortingEdits:
    """Editing sorting operations."""

    def test_edit_sort_key(self, nb_runner):
        """Edit the key function for sorting."""
        nb_runner.create_notebook(
            [
                "data = [('b', 2), ('a', 3), ('c', 1)]  # sort key source",
                "result = sorted(data, key=lambda x: x[0])\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [('a', 3), ('b', 2), ('c', 1)]" in nb_runner.get_output(2)

        # Sort by second element
        nb_runner.set_cell_source(2, "result = sorted(data, key=lambda x: x[1])\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [('c', 1), ('b', 2), ('a', 3)]" in nb_runner.get_output(2)

    def test_edit_sort_reverse(self, nb_runner):
        """Toggle reverse sort."""
        nb_runner.create_notebook(
            [
                "nums = [3, 1, 4, 1, 5, 9]  # sort reverse source",
                "result = sorted(nums)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 1, 3, 4, 5, 9]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = sorted(nums, reverse=True)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [9, 5, 4, 3, 1, 1]" in nb_runner.get_output(2)

    def test_edit_sort_source_data(self, nb_runner):
        """Edit source data, verify sort propagates."""
        nb_runner.create_notebook(
            [
                "words = ['banana', 'apple', 'cherry']  # sort data source",
                "result = sorted(words)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['apple', 'banana', 'cherry']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "words = ['zebra', 'mango', 'fig']  # sort data source v2")
        nb_runner.run_all()
        assert "result = ['fig', 'mango', 'zebra']" in nb_runner.get_output(2)

    def test_sort_then_slice(self, nb_runner):
        """Sort then take a slice, edit the slice."""
        nb_runner.create_notebook(
            [
                "vals = [50, 20, 80, 10, 90, 40]  # sort slice source",
                "sorted_vals = sorted(vals)",
                "top3 = sorted_vals[-3:]\nprint(f'top3 = {top3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3 = [50, 80, 90]" in nb_runner.get_output(3)

        # Change to bottom 2
        nb_runner.set_cell_source(3, "top3 = sorted_vals[:2]\nprint(f'top3 = {top3}')")
        nb_runner.run_all()
        assert "top3 = [10, 20]" in nb_runner.get_output(3)


# Sorting with custom key functions interaction tests.
# Tests that editing sort keys or comparison functions properly
# invalidates sorted outputs downstream.
@pytest.mark.integration
class TestSortingKeyInteraction:
    """Test sorting with custom keys and cache invalidation."""

    def test_sort_by_key_edit(self, nb_runner):
        """Editing sort key function should propagate."""
        nb_runner.create_notebook(
            [
                "items = [('banana', 3), ('apple', 1), ('cherry', 2)]",
                "def sort_key(item):\n    return item[0]",
                "sorted_items = sorted(items, key=sort_key)",
                "result = [x[0] for x in sorted_items]",
                "print(f'result={result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "data = [5, 3, 8, 1, 9]",
                "ascending = True",
                "ordered = sorted(data, reverse=not ascending)",
                "print(f'ordered={ordered}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "records = [('Alice', 85), ('Bob', 92), ('Charlie', 85), ('David', 92)]",
                "def multi_key(r):\n    return (-r[1], r[0])",
                "ranked = sorted(records, key=multi_key)",
                "names = [r[0] for r in ranked]",
                "print(f'names={names}')",
            ]
        )
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


class TestSortedKeyFunctions:
    """built-in sorted with key functions."""

    def test_sorted_len_key(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['banana', 'pie', 'strawberry', 'kiwi']",
                "by_len = sorted(words, key=len)\nby_last = sorted(words, key=lambda w: w[-1])\nprint(f'by_len={by_len} by_last={by_last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_len=['pie', 'kiwi', 'banana', 'strawberry']" in out

    def test_sorted_multi_key(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [('Bob', 85), ('Alice', 90), ('Charlie', 85), ('Alice', 80)]",
                "ordered = sorted(data, key=lambda x: (x[1], x[0]))\nprint(f'ordered={ordered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('Alice', 80)" in out

    def test_sorted_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [3, 1, 4, 1, 5, 9, 2, 6]",
                "asc = sorted(items)\ndesc = sorted(items, reverse=True)\nprint(f'asc={asc} desc={desc}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "asc=[1, 1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "items = [10, 30, 20, 50, 40]")
        nb_runner.run_all()
        assert "asc=[10, 20, 30, 40, 50]" in nb_runner.get_output(2)


class TestSortedMultiKeyReverse:
    """sorted with multiple keys and reverse."""

    def test_multi_key_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [('Alice', 30), ('Bob', 25), ('Carol', 30), ('Dave', 25)]",
                "by_age_name = sorted(data, key=lambda x: (x[1], x[0]))\nnames = [n for n, a in by_age_name]\nprint(f'names={names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Bob', 'Dave', 'Alice', 'Carol']" in nb_runner.get_output(2)

    def test_reverse_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = [5, 2, 8, 1, 9, 3]",
                "asc = sorted(nums)\ndesc = sorted(nums, reverse=True)\nprint(f'asc={asc} desc={desc}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "asc=[1, 2, 3, 5, 8, 9]" in out
        assert "desc=[9, 8, 5, 3, 2, 1]" in out

    def test_sort_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = ['banana', 'apple', 'cherry']",
                "result = sorted(items)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['apple', 'banana', 'cherry']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "items = ['zebra', 'mango', 'fig']")
        nb_runner.run_all()
        assert "result=['fig', 'mango', 'zebra']" in nb_runner.get_output(2)
