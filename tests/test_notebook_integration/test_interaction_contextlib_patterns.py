"""Batch 381: contextlib.suppress and contextlib patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextlibPatterns:
    def test_suppress(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import suppress\ndata = {'a': 1}",
            "with suppress(KeyError):\n    val = data['missing']\nresult = data.get('a', 0)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=1" in nb_runner.get_output(2)

    def test_redirect_stdout(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import redirect_stdout\nimport io",
            "f = io.StringIO()\nwith redirect_stdout(f):\n    print('captured')\noutput = f.getvalue().strip()\nprint(f'output={output}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "output=captured" in nb_runner.get_output(2)

    def test_custom_contextmanager(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import contextmanager\n@contextmanager\ndef timer_mock():\n    log = ['enter']\n    yield log\n    log.append('exit')",
            "with timer_mock() as log:\n    log.append('work')\nprint(f'log={log}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['enter', 'work', 'exit']" in nb_runner.get_output(2)
