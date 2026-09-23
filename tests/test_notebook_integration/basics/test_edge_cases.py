"""Boundary cases: empty and whitespace-only cells, odd names, huge output, pickling."""

import textwrap

import pytest


# Edge case interaction tests.
#
# Tests empty cells, whitespace-only changes, very large output,
# cell reordering scenarios, and other boundary conditions.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestWhitespaceEdits:
    """Whitespace-only cell edits."""

    def test_add_trailing_newline(self, nb_runner):
        """Adding trailing newline should not invalidate cache."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

        # Add trailing newlines — should still work
        nb_runner.set_cell_source(1, "x = 42\n\n")
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_add_comment_only(self, nb_runner):
        """Adding a comment changes the code hash → recomputes."""
        nb_runner.create_notebook(
            [
                "val = 10",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        # Add a comment — semantically identical but different hash
        nb_runner.set_cell_source(1, "# Important value\nval = 10")
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestLargeOutput:
    """Large output scenarios."""

    def test_large_list_output(self, nb_runner):
        """Generate a large list, edit the size."""
        nb_runner.create_notebook(
            [
                "n = 100  # list size",
                "data = list(range(n))\nprint(f'len = {len(data)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len = 100" in nb_runner.get_output(2)

        # Make it bigger
        nb_runner.set_cell_source(1, "n = 1000  # list size bigger")
        nb_runner.run_all()
        assert "len = 1000" in nb_runner.get_output(2)

    def test_large_string_output(self, nb_runner):
        """Generate a large string, then edit pattern."""
        nb_runner.create_notebook(
            [
                "pattern = 'ab'  # string pattern",
                "big = pattern * 500\nprint(f'length = {len(big)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length = 1000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "pattern = 'xyz'  # string pattern changed")
        nb_runner.run_all()
        assert "length = 1500" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultipleOutputsPerCell:
    """Cells producing multiple variables."""

    def test_multi_output_edit_one(self, nb_runner):
        """Cell producing multiple vars, edit to change one."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2\nc = 3",
                "total = a + b + c\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Change one variable
        nb_runner.set_cell_source(1, "a = 100\nb = 2\nc = 3")
        nb_runner.run_all()
        assert "total = 105" in nb_runner.get_output(2)

    def test_swap_variable_assignments(self, nb_runner):
        """Swap which variables get which values."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = 20",
                "diff = x - y\nprint(f'diff = {diff}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "diff = -10" in nb_runner.get_output(2)

        # Swap values
        nb_runner.set_cell_source(1, "x = 20\ny = 10")
        nb_runner.run_all()
        assert "diff = 10" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRerunPatterns:
    """Patterns of re-running cells."""

    def test_run_same_cell_twice(self, nb_runner):
        """Run a cell twice without edits — idempotent."""
        nb_runner.create_notebook(
            [
                "x = 5  # initial",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Re-run without edits
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

    def test_edit_then_revert(self, nb_runner):
        """Edit a cell, run, then revert and run again."""
        nb_runner.create_notebook(
            [
                "val = 'original'  # version 1",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = original" in nb_runner.get_output(2)

        # Edit
        nb_runner.set_cell_source(1, "val = 'modified'  # version 2")
        nb_runner.run_all()
        assert "val = modified" in nb_runner.get_output(2)

        # Revert to original (note: different comment to avoid identical cell ambiguity)
        nb_runner.set_cell_source(1, "val = 'original'  # version 3 reverted")
        nb_runner.run_all()
        assert "val = original" in nb_runner.get_output(2)


# Extreme edge cases, working directory changes, pickling,
# subprocess interactions, time-sensitive patterns, and multi-cell class hierarchies.
#
# These tests target the deepest corners of the caching system.
@pytest.mark.integration
@pytest.mark.timeout(30)
class TestWorkingDirectoryChanges:
    """Test caching when working directory changes between cells."""

    @pytest.mark.files
    def test_chdir_and_relative_file_read(self, nb_runner, tmp_path):
        """Change working directory then read file with relative path."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        data_file = subdir / "data.txt"
        data_file.write_text("hello from subdir", encoding="utf-8")
        subdir_str = str(subdir).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import os\nos.chdir('{subdir_str}')",
                "with open('data.txt') as f:\n    content = f.read()",
                "print(f'Content: {content}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Content: hello from subdir" in out

    @pytest.mark.files
    def test_chdir_back_and_forth(self, nb_runner, tmp_path):
        """Change to directory, read file, change back."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.txt").write_text("from dir1", encoding="utf-8")
        (dir2 / "b.txt").write_text("from dir2", encoding="utf-8")
        dir1_str = str(dir1).replace("\\", "/")
        dir2_str = str(dir2).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import os\nos.chdir('{dir1_str}')\nwith open('a.txt') as f:\n    a = f.read()",
                f"os.chdir('{dir2_str}')\nwith open('b.txt') as f:\n    b = f.read()",
                "print(f'a={a}, b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "a=from dir1" in out
        assert "b=from dir2" in out


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
