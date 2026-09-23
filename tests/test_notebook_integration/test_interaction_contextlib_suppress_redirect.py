"""contextlib suppress and redirect_stdout."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextlibSuppressRedirect:
    def test_suppress(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress",
                "with suppress(KeyError):\n    d = {}\n    val = d['missing']\nresult = 'survived'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=survived" in nb_runner.get_output(2)

    def test_suppress_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress\ndata = [1, 2, 3]",
                "with suppress(IndexError):\n    val = data[10]\nresult = len(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from contextlib import suppress\ndata = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "result=5" in nb_runner.get_output(2)
