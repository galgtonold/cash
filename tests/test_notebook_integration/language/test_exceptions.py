"""Raising, catching and chaining exceptions across cells."""

import textwrap

import pytest
from nbclient.exceptions import CellExecutionError


# Exception handling code + cell edit interaction tests.
#
# Tests that exercise try/except blocks, error recovery code paths,
# and how cash handles errors and recovers across cell edits.
class TestTryExceptEdits:
    """Try/except blocks + cell edits."""

    @pytest.mark.core
    @pytest.mark.stress
    @pytest.mark.timeout(30)
    def test_fix_error_in_cell(self, nb_runner):
        """Cell has an error, fix it, re-run."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x / 0\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

        # Fix the error
        nb_runner.set_cell_source(2, "y = x / 2\nprint(f'y = {y}')")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 5.0" in nb_runner.get_output(2)

    @pytest.mark.core
    @pytest.mark.stress
    @pytest.mark.timeout(30)
    def test_edit_except_handler(self, nb_runner):
        """Edit the except handler logic."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "try:\n    val = data[10]\nexcept IndexError:\n    val = 'out of range'\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = out of range" in nb_runner.get_output(2)

        # Change to a valid index
        nb_runner.set_cell_source(
            2,
            "try:\n    val = data[1]\nexcept IndexError:\n    val = 'out of range'\nprint(f'val = {val}')",
        )
        nb_runner.run_all()
        assert "val = 2" in nb_runner.get_output(2)

    # Try/except/finally interaction tests.
    #
    # Tests editing code within try/except blocks, changing exception
    # types, and modifying finally clauses.
    @pytest.mark.stress
    @pytest.mark.control
    @pytest.mark.timeout(90)
    def test_edit_try_body(self, nb_runner):
        """Edit the code inside a try block."""
        nb_runner.create_notebook(
            [
                "x = 10  # divisor",
                "try:\n    result = 100 // x\nexcept ZeroDivisionError:\n    result = -1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Make it divide by zero
        nb_runner.set_cell_source(1, "x = 0  # divisor zero")
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.control
    @pytest.mark.timeout(90)
    def test_edit_the_fallback_in_the_except_handler(self, nb_runner):
        """Edit the except handler to return a different fallback."""
        nb_runner.create_notebook(
            [
                "data = 'not_a_number'  # bad data",
                "try:\n    val = int(data)\nexcept ValueError:\n    val = 0\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 0" in nb_runner.get_output(2)

        # Change fallback value
        nb_runner.set_cell_source(
            2,
            "try:\n    val = int(data)\nexcept ValueError:\n    val = -999\nprint(f'val = {val}')",
        )
        nb_runner.run_all()
        assert "val = -999" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.control
    @pytest.mark.timeout(90)
    def test_fix_error_then_rerun(self, nb_runner):
        """Fix code that was raising an exception."""
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 3]  # data list",
                "try:\n    val = nums[10]\nexcept IndexError:\n    val = 'out_of_bounds'\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = out_of_bounds" in nb_runner.get_output(2)

        # Fix the index
        nb_runner.set_cell_source(
            2,
            "try:\n    val = nums[2]\nexcept IndexError:\n    val = 'out_of_bounds'\nprint(f'val = {val}')",
        )
        nb_runner.run_all()
        assert "val = 3" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestErrorRecoveryChain:
    """Error in middle of chain, fix and propagate."""

    def test_error_in_middle_fix_and_continue(self, nb_runner):
        """Error in middle cell, fix it, run the rest."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 'string'  # TypeError",
                "c = b * 2\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="TypeError"):
            nb_runner.run_all()

        # Fix the error
        nb_runner.set_cell_source(2, "b = a + 5")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(3)

    def test_introduce_error_then_fix(self, nb_runner):
        """Working code → introduce error → fix it."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 21" in nb_runner.get_output(3)

        # Introduce an error
        nb_runner.set_cell_source(2, "y = undefined_var * 2")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

        # Fix it back
        nb_runner.set_cell_source(2, "y = x * 3")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 31" in nb_runner.get_output(3)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestConditionalErrorHandling:
    """Conditional error handling + edits."""

    def test_conditional_with_error_branch(self, nb_runner):
        """Edit condition to switch between error and success paths."""
        nb_runner.create_notebook(
            [
                "mode = 'safe'",
                "if mode == 'safe':\n    result = 42\nelse:\n    result = 1 / 0\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        # Switch to unsafe mode
        nb_runner.set_cell_source(1, "mode = 'unsafe'")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

        # Back to safe
        nb_runner.set_cell_source(1, "mode = 'safe'")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

    def test_guard_clause_edit(self, nb_runner):
        """Edit guard clause that prevents errors."""
        nb_runner.create_notebook(
            [
                "values = [1, 2, 0, 4]",
                "safe = [v for v in values if v != 0]\nresult = sum(10 / v for v in safe)\nprint(f'result = {result:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10/1 + 10/2 + 10/4 = 10 + 5 + 2.5 = 17.5
        assert "result = 17.50" in nb_runner.get_output(2)

        # Edit to remove the zero
        nb_runner.set_cell_source(1, "values = [1, 2, 5, 4]")
        nb_runner.run_all()
        # 10/1 + 10/2 + 10/5 + 10/4 = 10 + 5 + 2 + 2.5 = 19.5
        assert "result = 19.50" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.control
@pytest.mark.timeout(90)
class TestFinallyEdits:
    """Editing finally clauses."""

    def test_edit_finally_action(self, nb_runner):
        """Edit what happens in a finally block."""
        nb_runner.create_notebook(
            [
                "status = 'unknown'  # init status",
                "try:\n    result = 42\nfinally:\n    status = 'done'\nprint(f'status = {status}, result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "status = done" in nb_runner.get_output(2)
        assert "result = 42" in nb_runner.get_output(2)

        # Change finally action
        nb_runner.set_cell_source(
            2,
            "try:\n    result = 42\nfinally:\n    status = 'complete'\nprint(f'status = {status}, result = {result}')",
        )
        nb_runner.run_all()
        assert "status = complete" in nb_runner.get_output(2)

    def test_add_finally_clause(self, nb_runner):
        """Add a finally clause to existing try/except."""
        nb_runner.create_notebook(
            [
                "cleanup_done = False  # cleanup flag",
                "try:\n    x = 100\nexcept Exception:\n    x = 0\nprint(f'x = {x}, cleanup = {cleanup_done}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)
        assert "cleanup = False" in nb_runner.get_output(2)

        # Add finally
        nb_runner.set_cell_source(
            2,
            "try:\n    x = 100\nexcept Exception:\n    x = 0\nfinally:\n    cleanup_done = True\nprint(f'x = {x}, cleanup = {cleanup_done}')",
        )
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)
        assert "cleanup = True" in nb_runner.get_output(2)


# Assertion and debugging print interaction tests.
#
# Tests editing assert statements, debug prints, and
# conditional debugging output.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestAssertEdits:
    """Editing assert statements."""

    def test_assert_pass_then_fail(self, nb_runner):
        """Assert passes, then edit to make it fail."""
        nb_runner.create_notebook(
            [
                "x = 10  # assert source",
                "assert x > 5, f'Expected x > 5, got {x}'\nprint(f'x = {x} (ok)')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 10 (ok)" in nb_runner.get_output(2)

        # Change x to make assert fail
        nb_runner.set_cell_source(1, "x = 3  # assert source fail")
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

    def test_edit_assert_condition(self, nb_runner):
        """Edit the assert condition."""
        nb_runner.create_notebook(
            [
                "val = 42  # assert cond source",
                "assert val == 42\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Tighten assertion
        nb_runner.set_cell_source(2, "assert val > 0 and val < 100\nprint(f'val = {val} (in range)')")
        nb_runner.run_all()
        assert "val = 42 (in range)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDebugPrintEdits:
    """Editing debug print patterns."""

    def test_edit_debug_format(self, nb_runner):
        """Edit the debug print format."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]  # debug print source",
                "print(f'len={len(data)} sum={sum(data)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3 sum=6" in nb_runner.get_output(2)

        # Change to more detailed format
        nb_runner.set_cell_source(2, "print(f'data={data} len={len(data)} min={min(data)} max={max(data)}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "data=[1, 2, 3]" in out
        assert "min=1" in out
        assert "max=3" in out

    def test_add_remove_debug_prints(self, nb_runner):
        """Edit output content between runs."""
        nb_runner.create_notebook(
            [
                "a = 5\nb = 10  # debug prints source",
                "c = a + b",
                "print(f'a={a} b={b} c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "a=5 b=10 c=15" in out

        # Change print format
        nb_runner.set_cell_source(3, "print(f'sum={c}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "sum=15" in out2


# Error handling interaction edit tests.
#
# Tests editing cells that change error-handling behavior: adding/removing
# try/except blocks, changing raised exceptions, etc.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestErrorHandlingEdits:
    """Editing error-handling patterns."""

    def test_edit_add_try_except(self, nb_runner):
        """Add a try/except to a cell that previously had no error handling."""
        nb_runner.create_notebook(
            [
                "data = {'a': 1, 'b': 2}",
                "val = data.get('c', -1)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = -1" in nb_runner.get_output(2)

        # Edit to use try/except instead of .get()
        nb_runner.set_cell_source(
            2, "try:\n    val = data['c']\nexcept KeyError:\n    val = 'missing'\nprint(f'val = {val}')"
        )
        nb_runner.run_all()
        assert "val = missing" in nb_runner.get_output(2)

    def test_edit_fix_error_to_success(self, nb_runner):
        """Edit a cell from one that raises to one that succeeds."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "try:\n    result = 10 / x\nexcept ZeroDivisionError:\n    result = float('inf')\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = inf" in nb_runner.get_output(2)

        # Fix the error by changing x
        nb_runner.set_cell_source(1, "x = 2")
        nb_runner.run_all()
        assert "result = 5.0" in nb_runner.get_output(2)

    def test_edit_change_default_value(self, nb_runner):
        """Edit the default/fallback value in error handling."""
        nb_runner.create_notebook(
            [
                "items = [10, 20, 30]",
                "try:\n    val = items[5]\nexcept IndexError:\n    val = 0\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 0" in nb_runner.get_output(2)

        # Change default value
        nb_runner.set_cell_source(
            2, "try:\n    val = items[5]\nexcept IndexError:\n    val = -999\nprint(f'val = {val}')"
        )
        nb_runner.run_all()
        assert "val = -999" in nb_runner.get_output(2)

    def test_edit_remove_error_condition(self, nb_runner):
        """Edit data so error condition no longer triggers."""
        nb_runner.create_notebook(
            [
                "values = []",
                "try:\n    avg = sum(values) / len(values)\nexcept ZeroDivisionError:\n    avg = 0\nprint(f'avg = {avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg = 0" in nb_runner.get_output(2)

        # Add data so division works
        nb_runner.set_cell_source(1, "values = [10, 20, 30]")
        nb_runner.run_all()
        assert "avg = 20" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCustomExceptionAttrs:
    """custom exception classes with attributes."""

    def test_custom_exception(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n        super().__init__(msg)",
                "try:\n    raise AppError(404, 'not found')\nexcept AppError as e:\n    result = f'{e.code}:{e.msg}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=404:not found" in nb_runner.get_output(2)

    def test_exception_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class DBError(Exception): pass\nclass ConnError(DBError): pass",
                "try:\n    raise ConnError('timeout')\nexcept DBError as e:\n    caught_type = type(e).__name__\n    is_conn = isinstance(e, ConnError)\nprint(f'type={caught_type} is_conn={is_conn}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type=ConnError" in nb_runner.get_output(2)
        assert "is_conn=True" in nb_runner.get_output(2)

    def test_exception_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class ValidationError(ValueError):\n    pass\nthreshold = 100",
                "try:\n    val = 150\n    if val > threshold:\n        raise ValidationError(f'too high: {val}')\n    result = 'ok'\nexcept ValidationError as e:\n    result = str(e)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=too high: 150" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "class ValidationError(ValueError):\n    pass\nthreshold = 200")
        nb_runner.run_all()
        assert "result=ok" in nb_runner.get_output(2)


# Error handling and exception propagation interaction tests.
# Tests that editing code that raises/catches exceptions properly
# invalidates downstream cells.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestExceptionHandlingInteraction:
    """Test exception handling patterns with cache invalidation."""

    def test_try_except_edit_exception_type(self, nb_runner):
        """Editing which exception is caught should propagate."""
        nb_runner.create_notebook(
            [
                "def risky(x):\n    if x == 0:\n        raise ValueError('zero')\n    return 100 // x",
                "try:\n    val = risky(0)\nexcept ValueError as e:\n    val = -1",
                "print(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "val=-1" in out

        # Change to non-zero (no exception)
        nb_runner.set_cell_source(2, "try:\n    val = risky(5)\nexcept ValueError as e:\n    val = -1")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "val=20" in out

    def test_custom_exception_edit(self, nb_runner):
        """Editing custom exception handling should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class AppError(Exception):\n"
                    "    def __init__(self, code, msg):\n"
                    "        self.code = code\n"
                    "        self.msg = msg"
                ),
                "def process(x):\n    if x < 0:\n        raise AppError(400, 'negative')\n    return x * 2",
                "try:\n    result = process(-5)\n    status = 'ok'\nexcept AppError as e:\n    result = 0\n    status = f'error:{e.code}'",
                "print(f'result={result},status={status}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=0,status=error:400" in out

        nb_runner.set_cell_source(
            3,
            "try:\n    result = process(10)\n    status = 'ok'\nexcept AppError as e:\n    result = 0\n    status = f'error:{e.code}'",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=20,status=ok" in out

    def test_finally_block_edit(self, nb_runner):
        """Editing code with try/finally should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "try:\n"
                    "    val = 100 // 5\n"
                    "    status = 'success'\n"
                    "except ZeroDivisionError:\n"
                    "    val = -1\n"
                    "    status = 'error'\n"
                    "finally:\n"
                    "    cleanup = True"
                ),
                "print(f'val={val},status={status},cleanup={cleanup}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "val=20" in out
        assert "status=success" in out
        assert "cleanup=True" in out

        nb_runner.set_cell_source(
            1,
            (
                "try:\n"
                "    val = 100 // 0\n"
                "    status = 'success'\n"
                "except ZeroDivisionError:\n"
                "    val = -1\n"
                "    status = 'error'\n"
                "finally:\n"
                "    cleanup = True"
            ),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "val=-1" in out
        assert "status=error" in out
        assert "cleanup=True" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestExceptionChaining:
    """multiple exception handling with else and chained raises."""

    def test_multiple_except(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def safe_parse(text):\n    try:\n        return int(text)\n    except ValueError:\n        return 'not_int'\n    except TypeError:\n        return 'not_str'",
                "r1 = safe_parse('42')\nr2 = safe_parse('abc')\nr3 = safe_parse(None)\nprint(f'r1={r1} r2={r2} r3={r3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=42 r2=not_int r3=not_str" in nb_runner.get_output(2)

    def test_try_else_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = {'key': 42}",
                "try:\n    val = data['key']\nexcept KeyError:\n    result = 'missing'\nelse:\n    result = f'found:{val}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=found:42" in nb_runner.get_output(2)
        # Edit to trigger exception
        nb_runner.set_cell_source(1, "data = {'other': 99}")
        nb_runner.run_all()
        assert "result=missing" in nb_runner.get_output(2)

    def test_exception_info(self, nb_runner):
        nb_runner.create_notebook(
            [
                "errors = []\nfor val in ['10', 'abc', '20', None]:\n    try:\n        errors.append(int(val))\n    except (ValueError, TypeError) as e:\n        errors.append(type(e).__name__)\nprint(f'errors={errors}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "errors=[10, 'ValueError', 20, 'TypeError']" in nb_runner.get_output(1)


# Exception hierarchy and custom exception interaction tests.
#
# Tests editing custom exception classes, raise patterns,
# and exception handling hierarchies.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestCustomExceptionEdits:
    """Editing custom exception classes."""

    def test_edit_custom_exception(self, nb_runner):
        """Edit a custom exception class."""
        nb_runner.create_notebook(
            [
                "class AppError(Exception):\n    def __init__(self, msg, code=0):\n        super().__init__(msg)\n        self.code = code",
                "try:\n    raise AppError('test', code=42)\nexcept AppError as e:\n    print(f'msg={e} code={e.code}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=test code=42" in nb_runner.get_output(2)

        # Add severity to exception
        nb_runner.set_cell_source(
            1,
            "class AppError(Exception):\n    def __init__(self, msg, code=0, severity='low'):\n        super().__init__(msg)\n        self.code = code\n        self.severity = severity",
        )
        nb_runner.set_cell_source(
            2,
            "try:\n    raise AppError('fail', code=99, severity='high')\nexcept AppError as e:\n    print(f'msg={e} code={e.code} sev={e.severity}')",
        )
        nb_runner.run_all()
        assert "msg=fail code=99 sev=high" in nb_runner.get_output(2)

    def test_edit_exception_hierarchy(self, nb_runner):
        """Edit exception hierarchy."""
        nb_runner.create_notebook(
            [
                "class BaseErr(Exception): pass\nclass ChildErr(BaseErr): pass",
                "try:\n    raise ChildErr('child')\nexcept BaseErr as e:\n    print(f'caught: {type(e).__name__}: {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "caught: ChildErr: child" in nb_runner.get_output(2)

        # Add new intermediate error
        nb_runner.set_cell_source(
            1,
            "class BaseErr(Exception): pass\nclass MidErr(BaseErr): pass\nclass ChildErr(MidErr): pass",
        )
        nb_runner.run_all()
        assert "caught: ChildErr: child" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestExceptionHandlingEdits:
    """Editing exception handling patterns."""

    def test_edit_except_clause(self, nb_runner):
        """Edit which exceptions are caught."""
        nb_runner.create_notebook(
            [
                "def risky(x):\n    if x == 0:\n        raise ValueError('zero')\n    return 10 / x",
                "try:\n    result = risky(0)\nexcept ValueError as e:\n    result = f'error: {e}'\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = error: zero" in nb_runner.get_output(2)

        # Change to catch Exception
        nb_runner.set_cell_source(
            2,
            "try:\n    result = risky(0)\nexcept Exception as e:\n    result = f'caught: {type(e).__name__}'\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = caught: ValueError" in nb_runner.get_output(2)


# Interaction test: exception hierarchy with custom base and derived.
# Tests custom exception hierarchy, isinstance checks, except chaining,
# and cross-cell error handling patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestExceptionHierarchyCustom:
    """Test custom exception hierarchy across cells."""

    def test_exception_hierarchy(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define exception hierarchy
                "class AppError(Exception):\n    '''Base application error.'''\n    pass\nclass ValidationError(AppError):\n    def __init__(self, field, message):\n        self.field = field\n        super().__init__(f'{field}: {message}')\nclass NotFoundError(AppError):\n    def __init__(self, resource):\n        self.resource = resource\n        super().__init__(f'{resource} not found')\nprint('Exception hierarchy defined')",
                # Cell 2: raise and catch
                "errors = []\ntry:\n    raise ValidationError('email', 'invalid format')\nexcept AppError as e:\n    errors.append(str(e))\ntry:\n    raise NotFoundError('User#42')\nexcept AppError as e:\n    errors.append(str(e))\nprint(f'caught={len(errors)}')\nfor e in errors:\n    print(f'  {e}')",
                # Cell 3: isinstance checks
                "v = ValidationError('name', 'too short')\nis_app = isinstance(v, AppError)\nis_exc = isinstance(v, Exception)\nis_notfound = isinstance(v, NotFoundError)\nprint(f'is_app={is_app}')\nprint(f'is_exc={is_exc}')\nprint(f'is_notfound={is_notfound}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class MyError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        super().__init__(f'[{code}] {msg}')\nprint('MyError defined')",
                "try:\n    raise MyError(404, 'Not Found')\nexcept MyError as e:\n    result = str(e)\n    code = e.code\nprint(f'result={result}')\nprint(f'code={code}')",
                "is_client = 400 <= code < 500\nprint(f'client_error={is_client}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[404] Not Found" in nb_runner.get_output(2)
        assert "client_error=True" in nb_runner.get_output(3)

        # Change error code
        nb_runner.set_cell_source(
            2,
            "try:\n    raise MyError(500, 'Internal Error')\nexcept MyError as e:\n    result = str(e)\n    code = e.code\nprint(f'result={result}')\nprint(f'code={code}')",
        )
        nb_runner.run_cells([2, 3])
        assert "result=[500] Internal Error" in nb_runner.get_output(2)
        assert "client_error=False" in nb_runner.get_output(3)

    def test_exception_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class DomainError(Exception):\n    pass\nclass OverflowDomainError(DomainError):\n    pass\nprint('defined')",
                "chain = [cls.__name__ for cls in OverflowDomainError.__mro__]\nprint(f'mro={chain}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "OverflowDomainError" in out
        assert "DomainError" in out
        assert "Exception" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "DomainError" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTryExceptElseFinally:
    """try except else finally patterns."""

    def test_exception_handling_flow(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "results = []\ntry:\n    x = 10 / 2\nexcept ZeroDivisionError:\n    results.append('except')\nelse:\n    results.append('else')\nfinally:\n    results.append('finally')\nprint(f'results={results} x={x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "results=['else', 'finally']" in out
        assert "x=5.0" in out

    def test_multiple_except(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "errors = []\nfor val in ['10', 'abc', None, '0']:\n    try:\n        result = 100 / int(val)\n    except (ValueError, TypeError) as e:\n        errors.append(type(e).__name__)\n    except ZeroDivisionError:\n        errors.append('ZeroDiv')\nprint(f'errors={errors}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "errors=['ValueError', 'TypeError', 'ZeroDiv']" in nb_runner.get_output(2)

    def test_exception_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "try:\n    val = int('abc')\n    msg = 'ok'\nexcept ValueError:\n    msg = 'error'\nprint(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=error" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "try:\n    val = int('42')\n    msg = 'ok'\nexcept ValueError:\n    msg = 'error'\nprint(f'msg={msg}')"
        )
        nb_runner.run_all()
        assert "msg=ok" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTryExceptFinally:
    """try/except/finally patterns with cell edits."""

    def test_try_except_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = {'a': 1, 'b': 2}",
                "try:\n    val = data['c']\nexcept KeyError:\n    val = -1\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=-1" in nb_runner.get_output(2)

    def test_try_except_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "numbers = [10, 0, 5]",
                "results = []\nfor n in numbers:\n    try:\n        results.append(100 // n)\n    except ZeroDivisionError:\n        results.append(-999)\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, -999, 20]" in nb_runner.get_output(2)
        # Edit data
        nb_runner.set_cell_source(1, "numbers = [5, 2, 0, 4]")
        nb_runner.run_all()
        assert "results=[20, 50, -999, 25]" in nb_runner.get_output(2)

    def test_try_finally_cleanup(self, nb_runner):
        nb_runner.create_notebook(
            [
                "log = []",
                "try:\n    log.append('start')\n    x = 42\n    log.append('done')\nfinally:\n    log.append('cleanup')\nprint(f'log={log} x={x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log=['start', 'done', 'cleanup']" in out
        assert "x=42" in out


# Exception handling & custom exceptions — cash caching with error patterns.
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


# Error handling patterns, type annotations, abstract classes,
# metaclass interactions, and exception flow caching.
#
# Tests how cash handles try/except, custom exceptions, type-annotated code,
# abstract base classes, metaclass-driven class creation, and exception
# propagation across cells.
@pytest.mark.integration
@pytest.mark.stress
class TestTypeAnnotations:
    """Test that type-annotated code caches correctly."""

    def test_typed_function(self, nb_runner):
        """Function with type annotations."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def add(x: int, y: int) -> int:
                    return x + y

                result: int = add(3, 4)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(1)


@pytest.mark.integration
@pytest.mark.stress
class TestStringProcessingPatterns:
    """Test complex string processing and regex patterns."""

    def test_format_spec_patterns(self, nb_runner):
        """Various format spec patterns."""
        nb_runner.create_notebook(
            [
                "value = 3.14159265",
                textwrap.dedent("""\
                results = [
                    f"{value:.2f}",
                    f"{value:.4e}",
                    f"{1000000:,}",
                    f"{0.75:.1%}",
                ]
                print(results)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "3.14" in output
        assert "1,000,000" in output
        assert "75.0%" in output


# Exception handling, try/except, custom exceptions, error propagation,
# and conditional error recovery across cells.
@pytest.mark.integration
@pytest.mark.stress
class TestExceptionHandlingCaching:
    """Test that exception handling patterns cache correctly."""

    def test_exception_change_propagation(self, nb_runner):
        """Change exception class → downstream catches update."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "code=404" in nb_runner.get_output(3)

        # Change exception class
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class MyError(Exception):
                def __init__(self, code, detail=""):
                    self.code = code
                    self.detail = detail
                    super().__init__(f"Error {code}: {detail}")
        """),
        )
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            def process():
                raise MyError(500, "server error")
        """),
        )
        nb_runner.set_cell_source(
            3,
            textwrap.dedent("""\
            try:
                process()
            except MyError as e:
                print(f"code={e.code} detail={e.detail}")
        """),
        )
        nb_runner.run_all()
        assert "code=500 detail=server error" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestMultiCellErrorRecovery:
    """Test error recovery spanning multiple cells."""

    def test_sentinel_value_propagation(self, nb_runner):
        """Sentinel values propagate correctly through cache."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 z_missing=True" in nb_runner.get_output(3)
