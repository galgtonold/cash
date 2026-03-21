"""
Batch 319: frozenset operations with caching.
Tests frozenset creation, set operations (union, intersection, difference), and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestFrozensetOps:
    """Test frozenset operation caching."""

    def test_frozenset_intersection(self, nb_runner):
        """Frozenset intersection, verify caching."""
        nb_runner.create_notebook([
            "a = frozenset([1, 2, 3, 4, 5])",
            "b = frozenset([3, 4, 5, 6, 7])",
            "common = a & b\nresult = sorted(common)",
            "print(f'common={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "common=[3, 4, 5]" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "common=[3, 4, 5]" in out2

    def test_frozenset_union_edit(self, nb_runner):
        """Edit one frozenset, verify union updates."""
        nb_runner.create_notebook([
            "s1 = frozenset(['a', 'b', 'c'])",
            "s2 = frozenset(['c', 'd', 'e'])",
            "merged = s1 | s2\ncount = len(merged)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=5" in out

        nb_runner.set_cell_source(2, "s2 = frozenset(['c', 'd', 'e', 'f'])")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=6" in out2

    def test_frozenset_as_dict_key(self, nb_runner):
        """Use frozensets as dict keys (hashable)."""
        nb_runner.create_notebook([
            "key1 = frozenset([1, 2])\nkey2 = frozenset([3, 4])",
            "lookup = {key1: 'first', key2: 'second'}",
            "query = frozenset([1, 2])\nval = lookup[query]\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "val=first" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "val=first" in out2
