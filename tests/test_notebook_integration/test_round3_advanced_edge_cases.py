"""
Advanced edge cases and uncommon Python patterns.

Tests focusing on:
1. Global/nonlocal keyword interactions
2. Type annotation patterns (no runtime effect)
3. Property decorators and descriptors
4. Dataclass patterns
5. Complex unpacking patterns
6. Generator/iterator patterns across cells
7. Complex string operations (multiline, raw, bytes)
8. Chained method calls
9. Boolean logic chains
10. Default argument patterns
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestPropertyAndDescriptorPatterns:
    """Tests for property decorators and descriptor protocol."""

    @pytest.mark.core
    def test_class_with_property(self, nb_runner):
        """Property decorator should work within caching."""
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def area(self):\n        import math\n        return math.pi * self._radius ** 2\n    @property\n    def radius(self):\n        return self._radius\n    @radius.setter\n    def radius(self, value):\n        self._radius = value",
                "c = Circle(5)",
                "area1 = round(c.area, 2)",
                "print(f'area1={area1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "area1=78.54" in output

        # Change class
        nb_runner.set_cell_source(
            1,
            "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def area(self):\n        import math\n        return math.pi * self._radius ** 2 * 2\n    @property\n    def radius(self):\n        return self._radius\n    @radius.setter\n    def radius(self, value):\n        self._radius = value",
        )
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "area1=157.08" in output2


class TestBooleanLogicChains:
    """Tests for complex boolean logic across cells."""

    @pytest.mark.core
    def test_boolean_chain_across_cells(self, nb_runner):
        """Boolean conditions computed across cells."""
        nb_runner.create_notebook(
            [
                "age = 25\nincome = 50000\ncredit_score = 720",
                "is_adult = age >= 18\nhas_income = income > 30000\ngood_credit = credit_score >= 700",
                "approved = is_adult and has_income and good_credit",
                "print(f'approved={approved}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "approved=True" in output

        # Change one condition to fail
        nb_runner.set_cell_source(1, "age = 16\nincome = 50000\ncredit_score = 720")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "approved=False" in output2


class TestDefaultArgumentPatterns:
    """Tests for default argument and mutable default patterns."""

    @pytest.mark.core
    def test_function_with_default_args(self, nb_runner):
        """Function with default arguments."""
        nb_runner.create_notebook(
            [
                "def greet(name, greeting='Hello', exclaim=True):\n    msg = f'{greeting}, {name}'\n    return msg + '!' if exclaim else msg",
                "r1 = greet('World')\nr2 = greet('Cash', 'Hi', False)",
                "print(f'r1={r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "r1=Hello, World!" in output
        assert "r2=Hi, Cash" in output


class TestMultilineStringPatterns:
    """Tests for multiline string and byte patterns."""

    @pytest.mark.core
    def test_multiline_string_formatting(self, nb_runner):
        """Multiline strings with formatting."""
        nb_runner.create_notebook(
            [
                "name = 'Cash'\nversion = '2.0'",
                "template = f'''Project: {name}\nVersion: {version}\nStatus: Active'''",
                "lines = template.split('\\n')\nline_count = len(lines)",
                "print(f'line_count={line_count} first={lines[0]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "line_count=3" in output
        assert "first=Project: Cash" in output


class TestComplexAssignmentPatterns:
    """Tests for augmented and complex assignment patterns."""

    @pytest.mark.core
    def test_conditional_expression_assignment(self, nb_runner):
        """Conditional expressions in assignments."""
        nb_runner.create_notebook(
            [
                "values = [3, 1, 4, 1, 5, 9, 2, 6]",
                "maximum = max(values)\nminimum = min(values)\nrange_val = maximum - minimum",
                "category = 'wide' if range_val > 5 else 'narrow'",
                "print(f'range={range_val} category={category}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "range=8" in output
        assert "category=wide" in output
