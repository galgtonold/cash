"""
Round 3 Batch 7: Extreme edge cases, working directory changes, pickling,
subprocess interactions, time-sensitive patterns, and multi-cell class hierarchies.

These tests target the deepest corners of the caching system.
"""
import pytest
import textwrap


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
        subdir_str = str(subdir).replace('\\', '/')

        nb_runner.create_notebook([
            f"import os\nos.chdir('{subdir_str}')",
            "with open('data.txt') as f:\n    content = f.read()",
            "print(f'Content: {content}')",
        ])
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
        dir1_str = str(dir1).replace('\\', '/')
        dir2_str = str(dir2).replace('\\', '/')

        nb_runner.create_notebook([
            f"import os\nos.chdir('{dir1_str}')\nwith open('a.txt') as f:\n    a = f.read()",
            f"os.chdir('{dir2_str}')\nwith open('b.txt') as f:\n    b = f.read()",
            "print(f'a={a}, b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "a=from dir1" in out
        assert "b=from dir2" in out


class TestPicklingAndSerialization:
    """Test that cached objects can be serialized correctly."""

    @pytest.mark.core
    def test_pickle_compatible_objects(self, nb_runner):
        """Objects that are pickle-compatible should cache correctly."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = {
                    'list': [1, 2, 3],
                    'dict': {'a': 1, 'b': 2},
                    'tuple': (10, 20),
                    'set_val': frozenset([1, 2, 3]),
                    'number': 42.5,
                    'string': 'hello',
                    'bool': True,
                    'none': None,
                }"""),
            "total = sum(data['list']) + data['number']",
            "print(f'Total: {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Total: 48.5" in out

    @pytest.mark.core
    def test_dataclass_caching(self, nb_runner):
        """Dataclass instances should be cached correctly."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass
                @dataclass
                class Point:
                    x: float
                    y: float
                    def distance(self):
                        return (self.x**2 + self.y**2)**0.5"""),
            "p = Point(3.0, 4.0)",
            "d = p.distance()\nprint(f'Distance: {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Distance: 5.0" in out


class TestClassHierarchyPatterns:
    """Test class inheritance and polymorphism across cells."""

    @pytest.mark.core
    def test_base_class_and_subclass(self, nb_runner):
        """Base class in one cell, subclass in another, usage in third."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Animal:
                    def __init__(self, name):
                        self.name = name
                    def speak(self):
                        return f'{self.name} says ...'"""),
            textwrap.dedent("""\
                class Dog(Animal):
                    def speak(self):
                        return f'{self.name} says Woof!'"""),
            textwrap.dedent("""\
                class Cat(Animal):
                    def speak(self):
                        return f'{self.name} says Meow!'"""),
            textwrap.dedent("""\
                animals = [Dog('Rex'), Cat('Whiskers'), Dog('Buddy')]
                for a in animals:
                    print(a.speak())"""),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Rex says Woof!" in out
        assert "Whiskers says Meow!" in out
        assert "Buddy says Woof!" in out

    @pytest.mark.core
    def test_change_base_class_invalidates_subclass(self, nb_runner):
        """Changing base class definition should invalidate subclass usage."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    def value(self):
                        return 10"""),
            textwrap.dedent("""\
                class Derived(Base):
                    def total(self):
                        return self.value() * 2"""),
            "d = Derived()\nresult = d.total()\nprint(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: 20" in out1

        # Change base class
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class Base:
                def value(self):
                    return 100"""))
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: 200" in out2


class TestComplexControlFlow:
    """Test complex control flow patterns."""

    @pytest.mark.control
    def test_try_except_with_fallback(self, nb_runner):
        """Try/except providing fallback value."""
        nb_runner.create_notebook([
            "data = {'key': 42}",
            textwrap.dedent("""\
                try:
                    value = data['missing_key']
                except KeyError:
                    value = -1"""),
            "print(f'Value: {value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Value: -1" in out

    @pytest.mark.control
    def test_comprehension_with_filter(self, nb_runner):
        """List comprehension with complex filter conditions."""
        nb_runner.create_notebook([
            "numbers = list(range(1, 31))",
            "fizzbuzz = [('FizzBuzz' if x%15==0 else 'Fizz' if x%3==0 else 'Buzz' if x%5==0 else str(x)) for x in numbers]",
            "print(f'FBs: {fizzbuzz.count(\"FizzBuzz\")}, Fizz: {fizzbuzz.count(\"Fizz\")}, Buzz: {fizzbuzz.count(\"Buzz\")}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "FBs: 2" in out
        assert "Fizz: 8" in out
        assert "Buzz: 4" in out

    @pytest.mark.control
    def test_nested_if_else(self, nb_runner):
        """Deeply nested if/else."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Category: medium" in out1

        # Change x to trigger different branch
        nb_runner.set_cell_source(1, "x = 25")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Category: high" in out2


class TestMultiStatementCellComplexity:
    """Test cells with many statements and complex internal dependencies."""

    @pytest.mark.core
    def test_cell_with_function_and_immediate_use(self, nb_runner):
        """Define a function and use it in the same cell."""
        nb_runner.create_notebook([
            "data = [3, 1, 4, 1, 5, 9, 2, 6]",
            textwrap.dedent("""\
                def quick_stats(lst):
                    n = len(lst)
                    mean = sum(lst) / n
                    variance = sum((x - mean)**2 for x in lst) / n
                    return mean, variance**0.5
                
                avg, std = quick_stats(data)"""),
            "print(f'Mean: {avg:.2f}, Std: {std:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Mean:" in out and "Std:" in out

    @pytest.mark.core
    def test_cell_with_import_and_computation(self, nb_runner):
        """Import and compute in the same cell."""
        nb_runner.create_notebook([
            "x = 2.0",
            "import math\ny = math.sqrt(x)\nz = math.log(x)",
            "print(f'sqrt: {y:.4f}, log: {z:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "sqrt: 1.4142" in out
        assert "log: 0.6931" in out

    @pytest.mark.core
    def test_complex_dictionary_comprehension(self, nb_runner):
        """Complex dict comprehension across cells."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c', 'd', 'e']",
            "values = [1, 2, 3, 4, 5]",
            "mapping = {k: v**2 for k, v in zip(keys, values) if v > 2}",
            "print(f'Mapping: {dict(sorted(mapping.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Mapping: {'c': 9, 'd': 16, 'e': 25}" in out


class TestNumpyPandasInteraction:
    """Test interaction between numpy and pandas across cells."""

    @pytest.mark.core
    def test_numpy_to_pandas_conversion(self, nb_runner):
        """Create numpy array, convert to pandas DataFrame."""
        nb_runner.create_notebook([
            "import numpy as np\nimport pandas as pd",
            "np.random.seed(0)\narr = np.random.randn(5, 3)",
            "df = pd.DataFrame(arr, columns=['x', 'y', 'z'])",
            "stats = df.describe().loc['mean'].round(2).to_dict()",
            "print(f'Means: {stats}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "Means:" in out

    @pytest.mark.core
    def test_pandas_apply_with_numpy(self, nb_runner):
        """Use numpy functions in pandas apply."""
        nb_runner.create_notebook([
            "import numpy as np\nimport pandas as pd",
            "df = pd.DataFrame({'x': [1, 4, 9, 16, 25]})",
            "df['sqrt_x'] = df['x'].apply(np.sqrt)",
            "print(df.to_string(index=False))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "1.0" in out and "5.0" in out


class TestSpecialVariablePatterns:
    """Test special Python variable patterns."""

    @pytest.mark.core
    def test_starred_assignment(self, nb_runner):
        """Starred unpacking in assignment."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "first, *rest, last = data",
            "print(f'First: {first}, Rest: {rest}, Last: {last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "First: 1, Rest: [2, 3, 4], Last: 5" in out

    @pytest.mark.core
    def test_multiple_return_values(self, nb_runner):
        """Function returning multiple values, unpacked in another cell."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def compute_stats(data):
                    return min(data), max(data), sum(data) / len(data)"""),
            "data = [10, 20, 30, 40, 50]",
            "lo, hi, avg = compute_stats(data)",
            "print(f'Min: {lo}, Max: {hi}, Avg: {avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Min: 10, Max: 50, Avg: 30.0" in out

    @pytest.mark.core
    def test_global_constant_pattern(self, nb_runner):
        """Constants defined early, used throughout."""
        nb_runner.create_notebook([
            "PI = 3.14159\nE = 2.71828\nTAU = PI * 2",
            "area = PI * 5**2",
            "growth = E ** 2",
            "print(f'Area: {area:.2f}, Growth: {growth:.2f}, TAU: {TAU:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Area: 78.54" in out
        assert "TAU: 6.28" in out


class TestFromImportEdgeCases:
    """Test edge cases specific to from-import patterns."""

    @pytest.mark.modules
    def test_from_import_multiple_names(self, nb_runner, tmp_path):
        """from module import name1, name2 — both should track the module."""
        mod = tmp_path / "multi_exports.py"
        mod.write_text(textwrap.dedent("""\
            def func_a():
                return 'A_v1'
            def func_b():
                return 'B_v1'
        """))
        tmp_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{tmp_str}')\nfrom multi_exports import func_a, func_b",
            "result = f'{func_a()}-{func_b()}'",
            "print(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: A_v1-B_v1" in out1

        # Change module
        mod.write_text(textwrap.dedent("""\
            def func_a():
                return 'A_v2'
            def func_b():
                return 'B_v2'
        """))
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: A_v2-B_v2" in out2

    @pytest.mark.modules
    def test_from_import_with_alias(self, nb_runner, tmp_path):
        """from module import name as alias — alias should track module."""
        mod = tmp_path / "aliased_mod.py"
        mod.write_text(textwrap.dedent("""\
            def compute():
                return 100
        """))
        tmp_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{tmp_str}')\nfrom aliased_mod import compute as calc",
            "result = calc()",
            "print(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: 100" in out1

        # Change module
        mod.write_text(textwrap.dedent("""\
            def compute():
                return 999
        """))
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: 999" in out2
