"""
Batch 306: Type conversion chain interaction tests.
Tests str→int, list→tuple→set, and dict→items→sorted conversion chains.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypeConvChainInteraction:
    """Test type conversion chains with cache invalidation."""

    def test_str_to_int_chain_edit(self, nb_runner):
        """Editing string input should propagate through int conversion."""
        nb_runner.create_notebook([
            "raw = '42'",
            "val = int(raw)",
            "doubled = val * 2",
            "print(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "doubled=84" in out

        nb_runner.set_cell_source(1, "raw = '100'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "doubled=200" in out

    def test_list_to_tuple_to_set_edit(self, nb_runner):
        """Editing list should propagate through tuple and set conversions."""
        nb_runner.create_notebook([
            "data = [3, 1, 4, 1, 5, 9, 2, 6]",
            "as_tuple = tuple(sorted(data))",
            "as_set = set(data)",
            "info = f'tuple_len={len(as_tuple)},set_len={len(as_set)}'",
            "print(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "tuple_len=8" in out
        assert "set_len=7" in out

        nb_runner.set_cell_source(1, "data = [1, 1, 1, 2, 2]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "tuple_len=5" in out
        assert "set_len=2" in out

    def test_dict_items_sorted_edit(self, nb_runner):
        """Editing dict should propagate through items/sorted chain."""
        nb_runner.create_notebook([
            "mapping = {'b': 2, 'a': 1, 'c': 3}",
            "items = list(mapping.items())",
            "sorted_items = sorted(items)",
            "keys = [k for k, v in sorted_items]",
            "print(f'keys={keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "keys=['a', 'b', 'c']" in out

        nb_runner.set_cell_source(1, "mapping = {'z': 26, 'x': 24, 'y': 25}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "keys=['x', 'y', 'z']" in out
