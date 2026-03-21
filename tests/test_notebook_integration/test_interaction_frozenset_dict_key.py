"""Batch 465: frozenset as dict key and set operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFrozensetDictKey:
    def test_frozenset_as_key(self, nb_runner):
        nb_runner.create_notebook([
            "a = frozenset({1, 2, 3})\nb = frozenset({3, 4, 5})",
            "d = {a: 'first', b: 'second'}\nresult = d[frozenset({1, 2, 3})]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=first" in nb_runner.get_output(2)

    def test_frozenset_set_ops(self, nb_runner):
        nb_runner.create_notebook([
            "a = frozenset({1, 2, 3})\nb = frozenset({2, 3, 4})",
            "u = sorted(a | b)\ni = sorted(a & b)\nd = sorted(a - b)\nprint(f'u={u} i={i} d={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "u=[1, 2, 3, 4]" in nb_runner.get_output(2)
        assert "i=[2, 3]" in nb_runner.get_output(2)
        assert "d=[1]" in nb_runner.get_output(2)

    def test_frozenset_edit(self, nb_runner):
        nb_runner.create_notebook([
            "fs = frozenset({10, 20, 30})",
            "contains = 20 in fs\ncount = len(fs)\nprint(f'contains={contains} count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "contains=True" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "fs = frozenset({40, 50})")
        nb_runner.run_all()
        assert "contains=False" in nb_runner.get_output(2)
        assert "count=2" in nb_runner.get_output(2)
