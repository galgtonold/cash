"""Batch 376: multiple exception handling with else and chained raises."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestExceptionChaining:
    def test_multiple_except(self, nb_runner):
        nb_runner.create_notebook([
            "def safe_parse(text):\n    try:\n        return int(text)\n    except ValueError:\n        return 'not_int'\n    except TypeError:\n        return 'not_str'",
            "r1 = safe_parse('42')\nr2 = safe_parse('abc')\nr3 = safe_parse(None)\nprint(f'r1={r1} r2={r2} r3={r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=42 r2=not_int r3=not_str" in nb_runner.get_output(2)

    def test_try_else_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = {'key': 42}",
            "try:\n    val = data['key']\nexcept KeyError:\n    result = 'missing'\nelse:\n    result = f'found:{val}'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=found:42" in nb_runner.get_output(2)
        # Edit to trigger exception
        nb_runner.set_cell_source(1, "data = {'other': 99}")
        nb_runner.run_all()
        assert "result=missing" in nb_runner.get_output(2)

    def test_exception_info(self, nb_runner):
        nb_runner.create_notebook([
            "errors = []\nfor val in ['10', 'abc', '20', None]:\n    try:\n        errors.append(int(val))\n    except (ValueError, TypeError) as e:\n        errors.append(type(e).__name__)\nprint(f'errors={errors}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "errors=[10, 'ValueError', 20, 'TypeError']" in nb_runner.get_output(1)
