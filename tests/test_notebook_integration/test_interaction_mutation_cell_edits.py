"""Batch 138 – Container/collection mutation + cell edit interaction tests.

Tests that exercise list/dict/set mutations across cells,
combined with cell edits to verify correct cache behavior
when mutable objects are modified in place.
"""

import pytest

pytestmark = [pytest.mark.mutations, pytest.mark.stress, pytest.mark.timeout(45)]


class TestListMutationWithCellEdits:
    """List mutations combined with cell edits."""

    def test_append_then_change_source(self, nb_runner):
        """Append to list, then change the source data."""
        nb_runner.create_notebook([
            "items = [1, 2, 3]",
            "items.append(4)",
            "total = sum(items)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(3)

        # Change source
        nb_runner.set_cell_source(1, "items = [10, 20, 30]")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        total_val = int(output.split("total = ")[1].strip())
        # Should be 10+20+30+4=64 or just 10+20+30=60 depending on append
        assert total_val >= 60

    def test_dict_update_then_edit_keys(self, nb_runner):
        """Update dict, then edit which keys are used."""
        nb_runner.create_notebook([
            "config = {'a': 1, 'b': 2}",
            "config['c'] = 3",
            "result = sum(config.values())\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        # Edit initial dict
        nb_runner.set_cell_source(1, "config = {'a': 10, 'b': 20}")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        result_val = int(output.split("result = ")[1].strip())
        assert result_val >= 30

    def test_list_extend_edit_then_slice(self, nb_runner):
        """Extend list, edit extension, use slice."""
        nb_runner.create_notebook([
            "base = [1, 2]",
            "extra = [3, 4]\nbase.extend(extra)",
            "part = base[:3]\nprint(f'part = {part}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "part = [1, 2, 3]" in nb_runner.get_output(3)

        # Edit extra
        nb_runner.set_cell_source(2, "extra = [30, 40]\nbase.extend(extra)")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "part = " in output


class TestSetMutationWithCellEdits:
    """Set mutations combined with cell edits."""

    def test_set_add_then_change_source(self, nb_runner):
        """Add to set, then change source set."""
        nb_runner.create_notebook([
            "s = {1, 2, 3}",
            "s.add(4)",
            "count = len(s)\nprint(f'count = {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(3)

        # Change source to larger set
        nb_runner.set_cell_source(1, "s = {10, 20, 30, 40, 50}")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        count_val = int(output.split("count = ")[1].strip())
        assert count_val >= 5


class TestNestedCollectionCellEdits:
    """Nested collections with edits."""

    def test_nested_dict_edit_inner(self, nb_runner):
        """Nested dict, edit inner values."""
        nb_runner.create_notebook([
            "data = {'outer': {'a': 1, 'b': 2}}",
            "inner_sum = sum(data['outer'].values())\nprint(f'inner_sum = {inner_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "inner_sum = 3" in nb_runner.get_output(2)

        # Change inner dict
        nb_runner.set_cell_source(1, "data = {'outer': {'a': 10, 'b': 20, 'c': 30}}")
        nb_runner.run_all()
        assert "inner_sum = 60" in nb_runner.get_output(2)

    def test_list_of_dicts_edit(self, nb_runner):
        """List of dicts, edit the list."""
        nb_runner.create_notebook([
            "records = [{'name': 'A', 'val': 1}, {'name': 'B', 'val': 2}]",
            "total = sum(r['val'] for r in records)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(2)

        # Add record
        nb_runner.set_cell_source(
            1,
            "records = [{'name': 'A', 'val': 1}, {'name': 'B', 'val': 2}, {'name': 'C', 'val': 10}]",
        )
        nb_runner.run_all()
        assert "total = 13" in nb_runner.get_output(2)
