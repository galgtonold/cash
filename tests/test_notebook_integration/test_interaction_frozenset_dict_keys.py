"""Batch 412: frozenset operations as dict keys and set algebra."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFrozensetDictKeys:
    def test_frozenset_basic(self, nb_runner):
        nb_runner.create_notebook([
            "fs = frozenset([1, 2, 3, 2, 1])",
            "size = len(fs)\nhas2 = 2 in fs\nhas5 = 5 in fs\nprint(f'size={size} has2={has2} has5={has5}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=3" in nb_runner.get_output(2)
        assert "has2=True" in nb_runner.get_output(2)
        assert "has5=False" in nb_runner.get_output(2)

    def test_frozenset_as_key(self, nb_runner):
        nb_runner.create_notebook([
            "groups = {}\ngroups[frozenset([1, 2])] = 'pair'\ngroups[frozenset([3, 4, 5])] = 'triple'",
            "r1 = groups[frozenset([2, 1])]\nr2 = groups[frozenset([5, 3, 4])]\nprint(f'r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=pair" in nb_runner.get_output(2)
        assert "r2=triple" in nb_runner.get_output(2)

    def test_frozenset_edit(self, nb_runner):
        nb_runner.create_notebook([
            "a = frozenset([1, 2, 3])\nb = frozenset([3, 4, 5])",
            "union = sorted(a | b)\ninter = sorted(a & b)\nprint(f'union={union} inter={inter}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "union=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "a = frozenset([10, 20])\nb = frozenset([20, 30])")
        nb_runner.run_all()
        assert "union=[10, 20, 30]" in nb_runner.get_output(2)
        assert "inter=[20]" in nb_runner.get_output(2)
