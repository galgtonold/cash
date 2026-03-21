"""Batch 341: try/except/finally patterns with cell edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTryExceptFinally:
    def test_try_except_basic(self, nb_runner):
        nb_runner.create_notebook([
            "data = {'a': 1, 'b': 2}",
            "try:\n    val = data['c']\nexcept KeyError:\n    val = -1\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=-1" in nb_runner.get_output(2)

    def test_try_except_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "numbers = [10, 0, 5]",
            "results = []\nfor n in numbers:\n    try:\n        results.append(100 // n)\n    except ZeroDivisionError:\n        results.append(-999)\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, -999, 20]" in nb_runner.get_output(2)
        # Edit data
        nb_runner.set_cell_source(1, "numbers = [5, 2, 0, 4]")
        nb_runner.run_all()
        assert "results=[20, 50, -999, 25]" in nb_runner.get_output(2)

    def test_try_finally_cleanup(self, nb_runner):
        nb_runner.create_notebook([
            "log = []",
            "try:\n    log.append('start')\n    x = 42\n    log.append('done')\nfinally:\n    log.append('cleanup')\nprint(f'log={log} x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log=['start', 'done', 'cleanup']" in out
        assert "x=42" in out
