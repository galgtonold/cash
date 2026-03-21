"""Batch 445: custom exception classes with attributes."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCustomExceptionAttrs:
    def test_custom_exception(self, nb_runner):
        nb_runner.create_notebook([
            "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n        super().__init__(msg)",
            "try:\n    raise AppError(404, 'not found')\nexcept AppError as e:\n    result = f'{e.code}:{e.msg}'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=404:not found" in nb_runner.get_output(2)

    def test_exception_chain(self, nb_runner):
        nb_runner.create_notebook([
            "class DBError(Exception): pass\nclass ConnError(DBError): pass",
            "try:\n    raise ConnError('timeout')\nexcept DBError as e:\n    caught_type = type(e).__name__\n    is_conn = isinstance(e, ConnError)\nprint(f'type={caught_type} is_conn={is_conn}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type=ConnError" in nb_runner.get_output(2)
        assert "is_conn=True" in nb_runner.get_output(2)

    def test_exception_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class ValidationError(ValueError):\n    pass\nthreshold = 100",
            "try:\n    val = 150\n    if val > threshold:\n        raise ValidationError(f'too high: {val}')\n    result = 'ok'\nexcept ValidationError as e:\n    result = str(e)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=too high: 150" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "class ValidationError(ValueError):\n    pass\nthreshold = 200")
        nb_runner.run_all()
        assert "result=ok" in nb_runner.get_output(2)
