"""
Extreme edge cases, working directory changes, pickling,
subprocess interactions, time-sensitive patterns, and multi-cell class hierarchies.

These tests target the deepest corners of the caching system.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestWorkingDirectoryChanges:
    """Test caching when working directory changes between cells."""

    @pytest.mark.files
    def test_chdir_and_relative_file_read(self, nb_runner, tmp_path):
        """Change working directory then read file with relative path."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        data_file = subdir / "data.txt"
        data_file.write_text("hello from subdir")
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
        (dir1 / "a.txt").write_text("from dir1")
        (dir2 / "b.txt").write_text("from dir2")
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
        """)
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
        """)
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
        """)
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
        """)
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: 999" in out2
