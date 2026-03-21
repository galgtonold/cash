"""Batch 510: try except else finally patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTryExceptElseFinally:
    def test_exception_handling_flow(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "results = []\ntry:\n    x = 10 / 2\nexcept ZeroDivisionError:\n    results.append('except')\nelse:\n    results.append('else')\nfinally:\n    results.append('finally')\nprint(f'results={results} x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "results=['else', 'finally']" in out
        assert "x=5.0" in out

    def test_multiple_except(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "errors = []\nfor val in ['10', 'abc', None, '0']:\n    try:\n        result = 100 / int(val)\n    except (ValueError, TypeError) as e:\n        errors.append(type(e).__name__)\n    except ZeroDivisionError:\n        errors.append('ZeroDiv')\nprint(f'errors={errors}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "errors=['ValueError', 'TypeError', 'ZeroDiv']" in nb_runner.get_output(2)

    def test_exception_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "try:\n    val = int('abc')\n    msg = 'ok'\nexcept ValueError:\n    msg = 'error'\nprint(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=error" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "try:\n    val = int('42')\n    msg = 'ok'\nexcept ValueError:\n    msg = 'error'\nprint(f'msg={msg}')")
        nb_runner.run_all()
        assert "msg=ok" in nb_runner.get_output(2)
