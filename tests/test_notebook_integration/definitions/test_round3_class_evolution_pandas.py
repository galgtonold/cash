"""
Multi-cell class evolution and refactoring patterns, complex
re-execution scenarios, out-of-order execution, and selective cell re-runs.

Tests patterns where users iteratively develop and refine code across cells,
re-run subsets of cells, and evolve class hierarchies during a session.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


# ============================================================
# Test Group 1: Class Evolution Patterns
# ============================================================


class TestClassEvolution:
    """Test iterative class development patterns common in notebooks."""

    def test_add_method_to_class(self, nb_runner):
        """Define class, use it, then add a method and re-use."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Calculator:
                    def __init__(self):
                        self.history = []
                    def add(self, a, b):
                        result = a + b
                        self.history.append(result)
                        return result
            """),
                textwrap.dedent("""\
                calc = Calculator()
                r1 = calc.add(3, 4)
                print(r1)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(2)

        # Add multiply method
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Calculator:
                def __init__(self):
                    self.history = []
                def add(self, a, b):
                    result = a + b
                    self.history.append(result)
                    return result
                def multiply(self, a, b):
                    result = a * b
                    self.history.append(result)
                    return result
        """),
        )
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            calc = Calculator()
            r1 = calc.add(3, 4)
            r2 = calc.multiply(5, 6)
            print(r1, r2)
        """),
        )
        nb_runner.run_all()
        assert "7 30" in nb_runner.get_output(2)

    def test_refine_function_signature(self, nb_runner):
        """Iteratively refine a function signature."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def process(data):
                    return sum(data)
            """),
                textwrap.dedent("""\
                result = process([1, 2, 3])
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)

        # Refine with default parameter
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def process(data, multiplier=1):
                return sum(data) * multiplier
        """),
        )
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            result = process([1, 2, 3], multiplier=10)
            print(result)
        """),
        )
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(2)


# ============================================================
# Test Group 2: Selective Cell Re-execution
# ============================================================


class TestSelectiveCellReexecution:
    """Test running subsets of cells (common notebook interaction pattern)."""

    def test_rerun_subset_of_cells(self, nb_runner):
        """Re-run only cells 2 and 3 out of 4."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a + 10",
                "c = b * 2",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(4)

        # Re-run cells 2-4
        nb_runner.run_cells([2, 3, 4])
        assert "30" in nb_runner.get_output(4)

    def test_modify_and_rerun_downstream(self, nb_runner):
        """Modify a cell and only re-run it and downstream cells."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1",
                "print(z)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3" in nb_runner.get_output(4)

        # Modify cell 2 and run cells 2-4
        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cells([2, 3, 4])
        assert "102" in nb_runner.get_output(4)


# ============================================================
# Test Group 3: Variable Reassignment Patterns
# ============================================================


class TestVariableReassignment:
    """Test variable reassignment and shadow patterns."""

    def test_progressive_refinement(self, nb_runner):
        """Progressively refine a variable across cells."""
        nb_runner.create_notebook(
            [
                "data = list(range(10))",
                "data = [x for x in data if x % 2 == 0]",  # filter evens
                "data = [x ** 2 for x in data]",  # square
                "print(data)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 4, 16, 36, 64]" in nb_runner.get_output(4)

    def test_conditional_variable_assignment(self, nb_runner):
        """Variable assigned conditionally across cells."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                "value = 75",
                textwrap.dedent("""\
                if value > threshold:
                    status = 'HIGH'
                else:
                    status = 'LOW'
                print(status)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HIGH" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "value = 25")
        nb_runner.run_all()
        assert "LOW" in nb_runner.get_output(3)


# ============================================================
# Test Group 4: Import Pattern Evolution
# ============================================================


# ============================================================
# Test Group 5: Datetime & Time-Based Patterns
# ============================================================


class TestDatetimePatterns:
    """Test datetime handling across cells."""

    def test_datetime_creation_and_formatting(self, nb_runner):
        """Create and format datetime objects across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timedelta",
                textwrap.dedent("""\
                start = datetime(2024, 1, 15, 10, 30)
                end = start + timedelta(days=7, hours=3)
            """),
                textwrap.dedent("""\
                diff = end - start
                formatted = end.strftime('%Y-%m-%d %H:%M')
                print(formatted, diff.days)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "2024-01-22 13:30" in output
        assert "7" in output

    def test_timedelta_arithmetic(self, nb_runner):
        """Timedelta arithmetic across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import timedelta",
                textwrap.dedent("""\
                work_day = timedelta(hours=8)
                work_week = work_day * 5
                work_month = work_week * 4
            """),
                textwrap.dedent("""\
                total_hours = work_month.total_seconds() / 3600
                print(int(total_hours))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "160" in nb_runner.get_output(3)
