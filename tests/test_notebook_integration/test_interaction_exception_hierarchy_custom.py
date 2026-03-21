"""
Interaction test: exception hierarchy with custom base and derived.
Tests custom exception hierarchy, isinstance checks, except chaining,
and cross-cell error handling patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestExceptionHierarchyCustom:
    """Test custom exception hierarchy across cells."""

    def test_exception_hierarchy(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define exception hierarchy
            "class AppError(Exception):\n    '''Base application error.'''\n    pass\nclass ValidationError(AppError):\n    def __init__(self, field, message):\n        self.field = field\n        super().__init__(f'{field}: {message}')\nclass NotFoundError(AppError):\n    def __init__(self, resource):\n        self.resource = resource\n        super().__init__(f'{resource} not found')\nprint('Exception hierarchy defined')",
            # Cell 2: raise and catch
            "errors = []\ntry:\n    raise ValidationError('email', 'invalid format')\nexcept AppError as e:\n    errors.append(str(e))\ntry:\n    raise NotFoundError('User#42')\nexcept AppError as e:\n    errors.append(str(e))\nprint(f'caught={len(errors)}')\nfor e in errors:\n    print(f'  {e}')",
            # Cell 3: isinstance checks
            "v = ValidationError('name', 'too short')\nis_app = isinstance(v, AppError)\nis_exc = isinstance(v, Exception)\nis_notfound = isinstance(v, NotFoundError)\nprint(f'is_app={is_app}')\nprint(f'is_exc={is_exc}')\nprint(f'is_notfound={is_notfound}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "caught=2" in out2
        assert "email: invalid format" in out2
        assert "User#42 not found" in out2
        out3 = nb_runner.get_output(3)
        assert "is_app=True" in out3
        assert "is_exc=True" in out3
        assert "is_notfound=False" in out3

    def test_exception_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class MyError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        super().__init__(f'[{code}] {msg}')\nprint('MyError defined')",
            "try:\n    raise MyError(404, 'Not Found')\nexcept MyError as e:\n    result = str(e)\n    code = e.code\nprint(f'result={result}')\nprint(f'code={code}')",
            "is_client = 400 <= code < 500\nprint(f'client_error={is_client}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[404] Not Found" in nb_runner.get_output(2)
        assert "client_error=True" in nb_runner.get_output(3)

        # Change error code
        nb_runner.set_cell_source(2, "try:\n    raise MyError(500, 'Internal Error')\nexcept MyError as e:\n    result = str(e)\n    code = e.code\nprint(f'result={result}')\nprint(f'code={code}')")
        nb_runner.run_cells([2, 3])
        assert "result=[500] Internal Error" in nb_runner.get_output(2)
        assert "client_error=False" in nb_runner.get_output(3)

    def test_exception_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class DomainError(Exception):\n    pass\nclass OverflowDomainError(DomainError):\n    pass\nprint('defined')",
            "chain = [cls.__name__ for cls in OverflowDomainError.__mro__]\nprint(f'mro={chain}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "OverflowDomainError" in out
        assert "DomainError" in out
        assert "Exception" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "DomainError" in nb_runner.get_output(2)
