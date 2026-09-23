"""
Collections patterns, enum usage, protocol/structural typing,
__slots__, and complex comprehension patterns.

Tests how cash handles specialized collection types, enums across cells,
Protocol-based structural subtyping, __slots__ classes, and deeply nested
comprehensions.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


# ============================================================
# Test Group 1: Collections Module Patterns
# ============================================================


# ============================================================
# Test Group 2: Enum Patterns
# ============================================================


class TestEnumPatterns:
    """Test enum usage across cells."""

    def test_enum_change_invalidation(self, nb_runner):
        """Changing enum definition should invalidate downstream."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from enum import Enum
                class Status(Enum):
                    ACTIVE = 'active'
                    INACTIVE = 'inactive'
            """),
                textwrap.dedent("""\
                s = Status.ACTIVE
                print(s.value)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "active" in nb_runner.get_output(2)

        # Change enum
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from enum import Enum
            class Status(Enum):
                ACTIVE = 'enabled'
                INACTIVE = 'disabled'
        """),
        )
        nb_runner.run_all()
        assert "enabled" in nb_runner.get_output(2)


# ============================================================
# Test Group 3: __slots__ Classes
# ============================================================


# ============================================================
# Test Group 4: Complex Comprehension Patterns
# ============================================================


class TestComplexComprehensions:
    """Test deeply nested and complex comprehension patterns."""

    def test_nested_list_comprehension(self, nb_runner):
        """Nested list comprehension with cross-cell dependency."""
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
                textwrap.dedent("""\
                flat = [x for row in matrix for x in row if x % 2 == 0]
                print(flat)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8]" in nb_runner.get_output(2)

    def test_generator_expression_materialized(self, nb_runner):
        """Generator expression consumed across cells."""
        nb_runner.create_notebook(
            [
                "numbers = range(1, 11)",
                textwrap.dedent("""\
                gen = (x**2 for x in numbers if x % 3 == 0)
                squares = list(gen)
                print(squares)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[9, 36, 81]" in nb_runner.get_output(2)


# ============================================================
# Test Group 5: Functional Programming Patterns
# ============================================================


class TestFunctionalPatterns:
    """Test functional programming patterns across cells."""

    def test_lru_cache_decorator(self, nb_runner):
        """lru_cache decorated function across cells."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache",
                textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def fib(n):
                    if n < 2:
                        return n
                    return fib(n-1) + fib(n-2)
            """),
                textwrap.dedent("""\
                result = fib(30)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "832040" in nb_runner.get_output(3)

    def test_higher_order_function_change(self, nb_runner):
        """Changing a higher-order function's component should invalidate."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def apply_twice(f, x):
                    return f(f(x))
            """),
                textwrap.dedent("""\
                def increment(x):
                    return x + 1
            """),
                textwrap.dedent("""\
                result = apply_twice(increment, 5)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            def increment(x):
                return x + 10
        """),
        )
        nb_runner.run_all()
        assert "25" in nb_runner.get_output(3)  # increment(increment(5)) = 5+10+10=25
