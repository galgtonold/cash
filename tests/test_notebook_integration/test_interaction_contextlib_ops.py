"""
Interaction test: contextlib contextmanager and suppress.
Tests custom context managers with @contextmanager decorator,
suppress() for exception handling, and cross-cell resource management.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextlibOps:
    """Test contextlib utilities across cells."""

    def test_contextmanager(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: custom context manager
            "from contextlib import contextmanager\nlog = []\n@contextmanager\ndef tracked(name):\n    log.append(f'enter:{name}')\n    yield name.upper()\n    log.append(f'exit:{name}')\n\nwith tracked('test') as val:\n    log.append(f'inside:{val}')\nlog_str = '->'.join(log)\nprint(f'flow={log_str}')",
            # Cell 2: check val from context
            "upper_val = val\nprint(f'upper_val={upper_val}')",
            # Cell 3: suppress exceptions
            "from contextlib import suppress\nresults = []\nfor x in [10, 0, 5]:\n    with suppress(ZeroDivisionError):\n        results.append(100 // x)\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "flow=enter:test->inside:TEST->exit:test" in out1
        out2 = nb_runner.get_output(2)
        assert "upper_val=TEST" in out2
        out3 = nb_runner.get_output(3)
        assert "results=[10, 20]" in out3

    def test_contextlib_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import contextmanager\n@contextmanager\ndef multiplier(factor):\n    yield lambda x: x * factor\n\nwith multiplier(3) as fn:\n    result = fn(10)\nprint(f'result={result}')",
            "doubled = result * 2\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(1)
        assert "doubled=60" in nb_runner.get_output(2)

        # Edit factor
        nb_runner.set_cell_source(1, "from contextlib import contextmanager\n@contextmanager\ndef multiplier(factor):\n    yield lambda x: x * factor\n\nwith multiplier(5) as fn:\n    result = fn(10)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "result=50" in nb_runner.get_output(1)
        assert "doubled=100" in nb_runner.get_output(2)

    def test_contextlib_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import suppress\nerrors = 0\nfor v in ['1', 'x', '3', 'y']:\n    with suppress(ValueError):\n        int(v)\n        continue\n    errors += 1\nprint(f'errors={errors}')",
            "has_errors = errors > 0\nprint(f'has_errors={has_errors}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "errors=2" in nb_runner.get_output(1)
        assert "has_errors=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_errors=True" in nb_runner.get_output(2)
