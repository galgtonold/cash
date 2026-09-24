"""Everyday constructs split over cells: classes, properties, imports, control flow and defaults."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestClassHierarchyPatterns:
    """Test class inheritance and polymorphism across cells."""

    @pytest.mark.core
    def test_change_base_class_invalidates_subclass(self, nb_runner):
        """Changing base class definition should invalidate subclass usage."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Base:
                    def value(self):
                        return 10"""),
                textwrap.dedent("""\
                class Derived(Base):
                    def total(self):
                        return self.value() * 2"""),
                "d = Derived()\nresult = d.total()\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: 20" in out1

        # Change base class
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Base:
                def value(self):
                    return 100"""),
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: 200" in out2


# Advanced edge cases and uncommon Python patterns.
#
# Tests focusing on:
# 1. Global/nonlocal keyword interactions
# 2. Type annotation patterns (no runtime effect)
# 3. Property decorators and descriptors
# 4. Dataclass patterns
# 5. Complex unpacking patterns
# 6. Generator/iterator patterns across cells
# 7. Complex string operations (multiline, raw, bytes)
# 8. Chained method calls
# 9. Boolean logic chains
# 10. Default argument patterns
@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestFromImportEdgeCases:
    """Test edge cases specific to from-import patterns."""

    @pytest.mark.modules
    def test_from_import_multiple_names(self, nb_runner, tmp_path):
        """from module import name1, name2 — both should track the module."""
        mod = tmp_path / "multi_exports.py"
        mod.write_text(
            textwrap.dedent("""\
            def func_a():
                return 'A_v1'
            def func_b():
                return 'B_v1'
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nfrom multi_exports import func_a, func_b",
                "result = f'{func_a()}-{func_b()}'",
                "print(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: A_v1-B_v1" in out1

        # Change module
        mod.write_text(
            textwrap.dedent("""\
            def func_a():
                return 'A_v2'
            def func_b():
                return 'B_v2'
        """),
            encoding="utf-8",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: A_v2-B_v2" in out2

    @pytest.mark.modules
    def test_from_import_with_alias(self, nb_runner, tmp_path):
        """from module import name as alias — alias should track module."""
        mod = tmp_path / "aliased_mod.py"
        mod.write_text(
            textwrap.dedent("""\
            def compute():
                return 100
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nfrom aliased_mod import compute as calc",
                "result = calc()",
                "print(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: 100" in out1

        # Change module
        mod.write_text(
            textwrap.dedent("""\
            def compute():
                return 999
        """),
            encoding="utf-8",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: 999" in out2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexControlFlow:
    """Test complex control flow patterns."""

    @pytest.mark.control
    def test_try_except_with_fallback(self, nb_runner):
        """Try/except providing fallback value."""
        nb_runner.create_notebook(
            [
                "data = {'key': 42}",
                textwrap.dedent("""\
                try:
                    value = data['missing_key']
                except KeyError:
                    value = -1"""),
                "print(f'Value: {value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Value: -1" in out

    @pytest.mark.control
    def test_comprehension_with_filter(self, nb_runner):
        """List comprehension with complex filter conditions."""
        nb_runner.create_notebook(
            [
                "numbers = list(range(1, 31))",
                "fizzbuzz = [('FizzBuzz' if x%15==0 else 'Fizz' if x%3==0 else 'Buzz' if x%5==0 else str(x)) for x in numbers]",
                'print(f\'FBs: {fizzbuzz.count("FizzBuzz")}, Fizz: {fizzbuzz.count("Fizz")}, Buzz: {fizzbuzz.count("Buzz")}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "FBs: 2" in out
        assert "Fizz: 8" in out
        assert "Buzz: 4" in out

    @pytest.mark.control
    def test_nested_if_else(self, nb_runner):
        """Deeply nested if/else."""
        nb_runner.create_notebook(
            [
                "x = 15",
                textwrap.dedent("""\
                if x > 20:
                    category = 'high'
                elif x > 10:
                    if x > 15:
                        category = 'medium-high'
                    else:
                        category = 'medium'
                else:
                    category = 'low'"""),
                "print(f'Category: {category}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Category: medium" in out1

        # Change x to trigger different branch
        nb_runner.set_cell_source(1, "x = 25")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Category: high" in out2


@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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
