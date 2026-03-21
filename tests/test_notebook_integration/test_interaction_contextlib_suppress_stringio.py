"""Batch 475: contextlib suppress and redirect to stringio."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextlibSuppressStringIO:
    def test_suppress_errors(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import suppress",
            "results = []\nfor key in ['a', 'b', 'c']:\n    d = {'a': 1, 'c': 3}\n    with suppress(KeyError):\n        results.append(d[key])\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 3]" in nb_runner.get_output(2)

    def test_redirect_stdout(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import redirect_stdout\nimport io",
            "buf = io.StringIO()\nwith redirect_stdout(buf):\n    print('captured line')\n    print('second line')\ncaptured = buf.getvalue()\nlines = captured.strip().split('\\n')\nprint(f'count={len(lines)} first={lines[0]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=2" in out
        assert "first=captured line" in out

    def test_suppress_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import suppress",
            "val = 0\nwith suppress(ZeroDivisionError):\n    val = 10 // 0\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=0" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "val = 0\nwith suppress(ZeroDivisionError):\n    val = 10 // 2\nprint(f'val={val}')")
        nb_runner.run_all()
        assert "val=5" in nb_runner.get_output(2)
