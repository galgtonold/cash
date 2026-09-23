"""Exception handling & custom exceptions — cash caching with error patterns."""

import textwrap

import pytest


@pytest.mark.stress
class TestCustomExceptions:
    """Test custom exception patterns across cells."""

    def test_exception_hierarchy(self, nb_runner):
        """Custom exception hierarchy across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class AppError(Exception):
                    def __init__(self, message, code=None):
                        super().__init__(message)
                        self.code = code

                class ValidationError(AppError):
                    pass

                class NotFoundError(AppError):
                    pass

                errors = []
                for cls, msg, code in [
                    (ValidationError, "Invalid input", 400),
                    (NotFoundError, "Item not found", 404),
                    (AppError, "Server error", 500),
                ]:
                    errors.append(cls(msg, code))
                print(f"count={len(errors)}")
            """),
                textwrap.dedent("""\
                for e in errors:
                    print(f"{type(e).__name__}: {e} (code={e.code})")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "ValidationError: Invalid input (code=400)" in out2
        assert "NotFoundError: Item not found (code=404)" in out2

    def test_exception_chaining(self, nb_runner):
        """Exception chaining across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class ProcessError(Exception):
                    pass

                def process(data):
                    try:
                        return int(data)
                    except ValueError as e:
                        raise ProcessError(f"Failed to process: {data}") from e

                results = []
                for item in ['42', 'abc', '99']:
                    try:
                        results.append(process(item))
                    except ProcessError as e:
                        results.append(str(e))
                print(f"results={results}")
            """),
                textwrap.dedent("""\
                nums = [r for r in results if isinstance(r, int)]
                print(f"valid_nums={nums} total={sum(nums)}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "42" in out1
        assert "Failed to process: abc" in out1
        assert "valid_nums=[42, 99] total=141" in nb_runner.get_output(2)


@pytest.mark.stress
class TestExceptionContextPatterns:
    """Test exception context patterns."""

    def test_exception_propagation(self, nb_runner):
        """Exception handling propagates on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                divisor = 2
            """),
                textwrap.dedent("""\
                try:
                    result = 100 / divisor
                    status = "ok"
                except ZeroDivisionError:
                    result = 0
                    status = "error"
                print(f"result={result} status={status}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50.0 status=ok" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            divisor = 0
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "result=0 status=error" in nb_runner.get_output(2)
