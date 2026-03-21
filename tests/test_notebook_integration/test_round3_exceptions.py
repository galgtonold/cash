"""
Batch 28: Exception handling, try/except, custom exceptions, error propagation,
and conditional error recovery across cells.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestExceptionHandlingCaching:
    """Test that exception handling patterns cache correctly."""

    def test_try_except_successful_path(self, nb_runner):
        """Try/except where no exception occurs."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def safe_divide(a, b):
                    try:
                        return a / b
                    except ZeroDivisionError:
                        return float('inf')
            """),
            textwrap.dedent("""\
                r1 = safe_divide(10, 3)
                r2 = safe_divide(10, 0)
                print(f"{r1:.4f} {r2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3.3333" in nb_runner.get_output(2)
        assert "inf" in nb_runner.get_output(2)

    def test_custom_exception_hierarchy(self, nb_runner):
        """Custom exception classes across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class AppError(Exception):
                    pass
                class ValidationError(AppError):
                    pass
                class NotFoundError(AppError):
                    pass
            """),
            textwrap.dedent("""\
                def validate(data):
                    if not data:
                        raise ValidationError("Empty data")
                    return data.upper()
            """),
            textwrap.dedent("""\
                try:
                    result = validate("hello")
                    print(f"ok: {result}")
                except ValidationError as e:
                    print(f"err: {e}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ok: HELLO" in nb_runner.get_output(3)

    def test_exception_change_propagation(self, nb_runner):
        """Change exception class → downstream catches update."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class MyError(Exception):
                    def __init__(self, code):
                        self.code = code
                        super().__init__(f"Error {code}")
            """),
            textwrap.dedent("""\
                def process():
                    raise MyError(404)
            """),
            textwrap.dedent("""\
                try:
                    process()
                except MyError as e:
                    print(f"code={e.code}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "code=404" in nb_runner.get_output(3)

        # Change exception class
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class MyError(Exception):
                def __init__(self, code, detail=""):
                    self.code = code
                    self.detail = detail
                    super().__init__(f"Error {code}: {detail}")
        """))
        nb_runner.set_cell_source(2, textwrap.dedent("""\
            def process():
                raise MyError(500, "server error")
        """))
        nb_runner.set_cell_source(3, textwrap.dedent("""\
            try:
                process()
            except MyError as e:
                print(f"code={e.code} detail={e.detail}")
        """))
        nb_runner.run_all()
        assert "code=500 detail=server error" in nb_runner.get_output(3)


class TestElseFinally:
    """Test try/except/else/finally patterns."""

    def test_try_else_finally(self, nb_runner):
        """Complete try/except/else/finally across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                log = []
                def do_work(val):
                    try:
                        result = 100 / val
                    except ZeroDivisionError:
                        log.append('except')
                        result = -1
                    else:
                        log.append('else')
                    finally:
                        log.append('finally')
                    return result
            """),
            textwrap.dedent("""\
                r1 = do_work(5)
                r2 = do_work(0)
                print(f"r1={r1} r2={r2} log={log}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "r1=20.0" in output
        assert "r2=-1" in output


class TestRetryPatterns:
    """Test retry/backoff patterns."""

    def test_retry_with_counter(self, nb_runner):
        """Retry logic across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class RetryCounter:
                    def __init__(self):
                        self.attempts = 0
                    def run(self, max_retries=3):
                        for i in range(max_retries):
                            self.attempts += 1
                            if i == max_retries - 1:
                                return "success"
                        return "fail"
            """),
            textwrap.dedent("""\
                rc = RetryCounter()
                result = rc.run(3)
                print(f"{result} attempts={rc.attempts}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "success attempts=3" in nb_runner.get_output(2)


class TestMultiCellErrorRecovery:
    """Test error recovery spanning multiple cells."""

    def test_guard_clause_pattern(self, nb_runner):
        """Guard clause validates data before processing."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def validate_data(data):
                    if not isinstance(data, list):
                        return None, "not a list"
                    if len(data) == 0:
                        return None, "empty"
                    return data, "ok"
            """),
            textwrap.dedent("""\
                data = [10, 20, 30]
                validated, status = validate_data(data)
            """),
            textwrap.dedent("""\
                if status == "ok":
                    total = sum(validated)
                    print(f"total={total}")
                else:
                    print(f"error: {status}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(3)

    def test_sentinel_value_propagation(self, nb_runner):
        """Sentinel values propagate correctly through cache."""
        nb_runner.create_notebook([
            "_MISSING = object()",
            textwrap.dedent("""\
                cache = {'a': 1, 'b': 2}
                def lookup(key):
                    return cache.get(key, _MISSING)
            """),
            textwrap.dedent("""\
                v1 = lookup('a')
                v2 = lookup('z')
                print(f"a={v1} z_missing={v2 is _MISSING}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 z_missing=True" in nb_runner.get_output(3)
