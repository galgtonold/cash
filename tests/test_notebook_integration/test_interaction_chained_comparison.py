"""Batch 392: chained comparison and identity operators."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainedComparison:
    def test_chained_compare(self, nb_runner):
        nb_runner.create_notebook([
            "x = 5",
            "in_range = 1 < x < 10\nresult = 'yes' if in_range else 'no'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=yes" in nb_runner.get_output(2)

    def test_chained_edit(self, nb_runner):
        nb_runner.create_notebook([
            "a, b, c = 1, 2, 3",
            "ascending = a < b < c\nresult = 'asc' if ascending else 'not'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=asc" in nb_runner.get_output(2)
        # Edit to break chain
        nb_runner.set_cell_source(1, "a, b, c = 1, 5, 3")
        nb_runner.run_all()
        assert "result=not" in nb_runner.get_output(2)

    def test_is_none_identity(self, nb_runner):
        nb_runner.create_notebook([
            "val = None\nother = 0\nempty = ''",
            "r1 = val is None\nr2 = other is None\nr3 = empty is not None\nprint(f'r1={r1} r2={r2} r3={r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=True r2=False r3=True" in nb_runner.get_output(2)
