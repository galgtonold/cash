"""
Round 3: Complex scenario integration tests.

Tests progressively complex caching interactions including:
- Deep dependency chains (5+ cells)
- Dataclass and complex object caching
- Function redefinition with downstream propagation
- Multi-variable assignment patterns
- Exception handling and recovery
- Cross-cell data transformations
- Global state interactions
- Nested function closures across cells
- Re-execution after code modifications
- Complex pandas operations
"""
import pytest

pytestmark = pytest.mark.core


class TestDeepDependencyChains:
    """Test deep dependency chains spanning many cells."""

    def test_six_cell_dependency_chain(self, nb_runner):
        """
        A chain of 6 cells where each depends on the previous.
        Modify the first cell and verify the entire chain updates.
        """
        nb_runner.create_notebook([
            # Cell 1: Base
            "base = 10",
            # Cell 2: Depends on base
            "step1 = base * 2",
            # Cell 3: Depends on step1
            "step2 = step1 + 5",
            # Cell 4: Depends on step2
            "step3 = step2 ** 2",
            # Cell 5: Depends on step3
            "step4 = step3 - 100",
            # Cell 6: Final output
            "print(f'Result: {step4}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # base=10 -> step1=20 -> step2=25 -> step3=625 -> step4=525
        out = nb_runner.get_output(6)
        assert "Result: 525" in out, f"Expected 525, got: {out}"

    def test_six_cell_chain_modification_propagates(self, nb_runner):
        """
        Run a 6-cell chain, modify cell 1, then re-run cell 6.
        The upstream simulation should detect the change and recompute.
        """
        nb_runner.create_notebook([
            "base = 10",
            "step1 = base * 2",
            "step2 = step1 + 5",
            "step3 = step2 ** 2",
            "step4 = step3 - 100",
            "print(f'Result: {step4}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out1 = nb_runner.get_output(6)
        assert "Result: 525" in out1

        # Modify the base cell
        nb_runner.set_cell_source(1, "base = 5")
        # Re-run from cell 1 through cell 6
        nb_runner.run_cells([1, 2, 3, 4, 5, 6])

        # base=5 -> step1=10 -> step2=15 -> step3=225 -> step4=125
        out2 = nb_runner.get_output(6)
        assert "Result: 125" in out2, f"Expected 125 after modification, got: {out2}"

    def test_branching_dependency_graph(self, nb_runner):
        """
        Test a diamond dependency pattern:
        Cell 1: x = 10
        Cell 2: a = x * 2   (depends on x)
        Cell 3: b = x + 5   (depends on x)
        Cell 4: c = a + b   (depends on a and b)
        """
        nb_runner.create_notebook([
            "x = 10",
            "a = x * 2",
            "b = x + 5",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(4)
        # x=10, a=20, b=15, c=35
        assert "c = 35" in out, f"Expected c=35, got: {out}"

    def test_branching_modify_root_all_branches_update(self, nb_runner):
        """
        Modify the root of a diamond and verify both branches update.
        """
        nb_runner.create_notebook([
            "x = 10",
            "a = x * 2",
            "b = x + 5",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "c = 35" in nb_runner.get_output(4)

        # Change x from 10 to 100
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cells([1, 2, 3, 4])

        # x=100, a=200, b=105, c=305
        out = nb_runner.get_output(4)
        assert "c = 305" in out, f"Expected c=305 after modification, got: {out}"


class TestDataclassCaching:
    """Test caching behavior with dataclass objects."""

    def test_dataclass_creation_and_use(self, nb_runner):
        """Test that dataclass instances are cached and reused correctly."""
        nb_runner.create_notebook([
            # Cell 1: Define dataclass
            """from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    name: str
    values: List[int] = field(default_factory=list)
    
    def total(self):
        return sum(self.values)""",
            # Cell 2: Create instance
            "cfg = Config(name='test', values=[1, 2, 3, 4, 5])",
            # Cell 3: Use it
            "print(f'Total: {cfg.total()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Total: 15" in out, f"Expected Total: 15, got: {out}"

    def test_dataclass_mutation_detection(self, nb_runner):
        """Test that mutating a dataclass field is detected."""
        nb_runner.create_notebook([
            """from dataclasses import dataclass, field
from typing import List

@dataclass
class Accumulator:
    items: List[int] = field(default_factory=list)
    
    def add(self, val):
        self.items.append(val)
        return self""",
            "acc = Accumulator()",
            "acc.add(10)\nacc.add(20)\nprint(f'Items: {acc.items}')",
            "total = sum(acc.items)\nprint(f'Total: {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out3 = nb_runner.get_output(3)
        assert "Items: [10, 20]" in out3, f"Expected [10, 20], got: {out3}"

        out4 = nb_runner.get_output(4)
        assert "Total: 30" in out4, f"Expected Total: 30, got: {out4}"


class TestFunctionRedefinition:
    """Test that redefining a function triggers downstream recomputation."""

    def test_redefine_function_invalidates_downstream(self, nb_runner):
        """
        Define a function in cell 1, use it in cell 2.
        Then redefine the function and re-run cell 2.
        """
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2",
            "result = transform(5)\nprint(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Result: 10" in nb_runner.get_output(2)

        # Redefine the function
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Result: 15" in out, f"Expected Result: 15 after redefine, got: {out}"

    def test_redefine_function_only_downstream_runs(self, nb_runner):
        """
        Redefine function in cell 1, run only downstream cell 2.
        The upstream simulation should detect the change and re-execute cell 1.
        """
        nb_runner.create_notebook([
            "def compute(x):\n    return x + 100",
            "val = compute(5)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "val = 105" in nb_runner.get_output(2)

        # Modify function definition
        nb_runner.set_cell_source(1, "def compute(x):\n    return x + 200")
        # Only run cell 2 - upstream should auto-execute cell 1
        nb_runner.run_cell(2)

        out = nb_runner.get_output(2)
        assert "val = 205" in out, f"Expected val=205, got: {out}"


class TestComplexAssignments:
    """Test complex assignment patterns."""

    def test_tuple_unpacking_multiple_targets(self, nb_runner):
        """Test tuple unpacking with many variables."""
        nb_runner.create_notebook([
            "data = (1, 'hello', [3, 4], {'a': 5})",
            "num, text, lst, dct = data",
            "print(f'{num} {text} {lst} {dct}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "1 hello [3, 4] {'a': 5}" in out, f"Got: {out}"

    def test_nested_tuple_unpacking(self, nb_runner):
        """Test nested tuple unpacking."""
        nb_runner.create_notebook([
            "pairs = [(1, 'a'), (2, 'b'), (3, 'c')]",
            "firsts = [x for x, y in pairs]\nseconds = [y for x, y in pairs]",
            "print(f'firsts={firsts}, seconds={seconds}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "firsts=[1, 2, 3]" in out, f"Got: {out}"
        assert "seconds=['a', 'b', 'c']" in out



class TestExceptionHandling:
    """Test caching behavior when cells raise exceptions."""


    def test_try_except_in_cell(self, nb_runner):
        """Test that try/except blocks are cached correctly."""
        nb_runner.create_notebook([
            "val = 'hello'",
            """try:
    result = int(val)
except ValueError:
    result = -1
print(f'result = {result}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = -1" in out, f"Got: {out}"


class TestCrossDataTransformations:
    """Test complex data transformations across cells."""

    def test_pandas_pipeline_across_cells(self, nb_runner):
        """Test a multi-cell pandas transformation pipeline."""
        nb_runner.create_notebook([
            # Cell 1: Create data
            """import pandas as pd
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
    'score': [85, 92, 78, 95, 88],
    'subject': ['math', 'math', 'math', 'science', 'science']
})""",
            # Cell 2: Group and aggregate
            "grouped = df.groupby('name')['score'].agg(['mean', 'max']).reset_index()",
            # Cell 3: Filter high performers
            "high_performers = grouped[grouped['mean'] >= 85]",
            # Cell 4: Final output
            "names = sorted(high_performers['name'].tolist())\nprint(f'High: {names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(4)
        # Alice: mean=90, Bob: mean=90, Charlie: mean=78
        assert "Alice" in out, f"Expected Alice, got: {out}"
        assert "Bob" in out, f"Expected Bob, got: {out}"

    def test_pandas_modify_source_data(self, nb_runner):
        """Modify source data and verify the pipeline updates."""
        nb_runner.create_notebook([
            """import pandas as pd
df = pd.DataFrame({'name': ['A', 'B'], 'val': [10, 20]})""",
            "total = df['val'].sum()\nprint(f'Total: {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Total: 30" in nb_runner.get_output(2)

        # Modify the DataFrame
        nb_runner.set_cell_source(1,
            "import pandas as pd\ndf = pd.DataFrame({'name': ['A', 'B', 'C'], 'val': [10, 20, 30]})")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Total: 60" in out, f"Expected Total: 60, got: {out}"

    def test_numpy_matrix_operations_across_cells(self, nb_runner):
        """Test numpy matrix operations across multiple cells."""
        nb_runner.create_notebook([
            "import numpy as np\nnp.random.seed(42)\nmatrix = np.random.randint(0, 100, (3, 4))",
            "row_sums = matrix.sum(axis=1)",
            "col_means = matrix.mean(axis=0)",
            "normalized = (matrix - col_means) / matrix.std(axis=0)",
            "print(f'Shape: {normalized.shape}, Mean of means: {normalized.mean():.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(5)
        assert "Shape: (3, 4)" in out, f"Got: {out}"


class TestGlobalStateInteractions:
    """Test how caching interacts with global state."""

    def test_global_list_mutation_across_cells(self, nb_runner):
        """Test that global list mutations are tracked correctly."""
        nb_runner.create_notebook([
            "data = []",
            "data.append(1)\ndata.append(2)",
            "data.append(3)",
            "print(f'data = {data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(4)
        assert "data = [1, 2, 3]" in out, f"Got: {out}"

    def test_dict_update_across_cells(self, nb_runner):
        """Test dict updates across multiple cells."""
        nb_runner.create_notebook([
            "config = {}",
            "config['host'] = 'localhost'\nconfig['port'] = 8080",
            "config['debug'] = True",
            "print(f'config = {config}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(4)
        assert "'host': 'localhost'" in out, f"Got: {out}"
        assert "'port': 8080" in out
        assert "'debug': True" in out


class TestNestedFunctions:
    """Test nested function definitions and closures across cells."""

    def test_closure_captures_cross_cell_variable(self, nb_runner):
        """
        Define a closure in one cell that captures a variable from another.
        """
        nb_runner.create_notebook([
            "factor = 3",
            "def make_multiplier(n):\n    return lambda x: x * n * factor",
            "mult = make_multiplier(2)\nresult = mult(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # 5 * 2 * 3 = 30
        out = nb_runner.get_output(3)
        assert "result = 30" in out, f"Got: {out}"

    def test_decorator_pattern_across_cells(self, nb_runner):
        """Test decorator defined in one cell, applied in another."""
        nb_runner.create_notebook([
            # Cell 1: Define decorator
            """def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper""",
            # Cell 2: Use decorator
            """@log_call
def greet(name):
    return f"Hello, {name}!"
    
msg = greet("World")
print(msg)""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "Hello, World!" in out, f"Got: {out}"


class TestReexecutionPatterns:
    """Test various re-execution patterns."""

    def test_skip_optimization_on_rerun(self, nb_runner):
        """
        Run all cells, then re-run them all.
        Second run should skip (or cache-hit) unchanged cells.
        """
        nb_runner.create_notebook([
            "x = 42",
            "y = x + 8\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "y = 50" in nb_runner.get_output(2)

        # Re-run all - should produce same result
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "y = 50" in out, f"Expected same result on re-run, got: {out}"


    def test_add_new_intermediate_dependency(self, nb_runner):
        """
        Run A -> C, then modify C to depend on new variable B.
        """
        nb_runner.create_notebook([
            "a = 10",
            "b = 99",
            "result = a * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 20" in nb_runner.get_output(3)

        # Now make result depend on b too
        nb_runner.set_cell_source(3, "result = a + b\nprint(f'result = {result}')")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "result = 109" in out, f"Expected result=109, got: {out}"


class TestComplexLoopPatterns:
    """Test complex loop patterns and their caching behavior."""

    def test_list_comprehension_with_function_call(self, nb_runner):
        """List comprehension calling a function defined in another cell."""
        nb_runner.create_notebook([
            "def process(x):\n    return x ** 2 + 1",
            "data = list(range(5))",
            "results = [process(x) for x in data]\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "results = [1, 2, 5, 10, 17]" in out, f"Got: {out}"

    def test_nested_loop_with_accumulator(self, nb_runner):
        """Nested loop accumulating results into a dict."""
        nb_runner.create_notebook([
            "categories = ['A', 'B']\nvalues = [1, 2, 3]",
            """results = {}
for cat in categories:
    results[cat] = []
    for val in values:
        results[cat].append(f'{cat}:{val}')
print(results)""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "'A': ['A:1', 'A:2', 'A:3']" in out, f"Got: {out}"
        assert "'B': ['B:1', 'B:2', 'B:3']" in out

    def test_while_loop_with_break(self, nb_runner):
        """While loop with break condition."""
        nb_runner.create_notebook([
            "threshold = 100",
            """total = 0
count = 0
while total < threshold:
    count += 1
    total += count
print(f'count={count}, total={total}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        # 1+2+3+...+13 = 91, +14 = 105 > 100
        assert "count=14, total=105" in out, f"Got: {out}"


class TestClassInheritancePatterns:
    """Test class inheritance and method caching."""

    def test_class_hierarchy_across_cells(self, nb_runner):
        """Define base class in one cell, derived in another, use in third."""
        nb_runner.create_notebook([
            # Cell 1: Base class
            """class Shape:
    def __init__(self, name):
        self.name = name
    def area(self):
        raise NotImplementedError""",
            # Cell 2: Derived class
            """class Circle(Shape):
    def __init__(self, radius):
        super().__init__('Circle')
        self.radius = radius
    def area(self):
        import math
        return math.pi * self.radius ** 2""",
            # Cell 3: Use
            "c = Circle(5)\nprint(f'{c.name}: area={c.area():.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Circle: area=78.54" in out, f"Got: {out}"

    def test_modify_base_class_invalidates_child(self, nb_runner):
        """Modifying base class should invalidate derived class usage."""
        nb_runner.create_notebook([
            """class Base:
    def greet(self):
        return "Hello"
""",
            """class Child(Base):
    def greet(self):
        return super().greet() + " World"
""",
            "obj = Child()\nprint(obj.greet())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Hello World" in nb_runner.get_output(3)

        # Modify the base class
        nb_runner.set_cell_source(1, """class Base:
    def greet(self):
        return "Hi"
""")
        nb_runner.run_cells([1, 2, 3])

        out = nb_runner.get_output(3)
        assert "Hi World" in out, f"Expected 'Hi World', got: {out}"


class TestMultiStatementCells:
    """Test cells with many statements."""

    def test_cell_with_20_statements(self, nb_runner):
        """Test a cell with many sequential statements."""
        nb_runner.create_notebook([
            # Cell 1: Many statements
            """a = 1
b = 2
c = a + b
d = c * 2
e = d - 1
f = e ** 2
g = f // 3
h = g + 10
i = h - 5
j = i * 3
k = j + 1
l = k - 2
m = l * 4
n = m // 5
o = n + 7
p = o - 3
q = p * 2
r = q + 1
s = r - 4
t = s * 3
print(f't = {t}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        # Let's compute: a=1, b=2, c=3, d=6, e=5, f=25, g=8, h=18, i=13,
        # j=39, k=40, l=38, m=152, n=30, o=37, p=34, q=68, r=69, s=65, t=195
        assert "t = 195" in out, f"Got: {out}"

    def test_mixed_assignments_and_expressions(self, nb_runner):
        """Test cells mixing assignments, print calls, and function definitions."""
        nb_runner.create_notebook([
            """data = [1, 2, 3, 4, 5]
total = sum(data)
mean = total / len(data)
above = [x for x in data if x > mean]
below = [x for x in data if x <= mean]
print(f'mean={mean}, above={above}, below={below}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "mean=3.0" in out, f"Got: {out}"
        assert "above=[4, 5]" in out
        assert "below=[1, 2, 3]" in out


class TestStringAndFormattingOperations:
    """Test string processing patterns."""

    def test_multiline_string_processing(self, nb_runner):
        """Test multiline string processing across cells."""
        nb_runner.create_notebook([
            'text = """Hello World\\nFoo Bar\\nBaz Qux"""',
            "lines = text.strip().split('\\n')",
            "upper_lines = [l.upper() for l in lines]",
            "result = ' | '.join(upper_lines)\nprint(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(4)
        assert "HELLO WORLD | FOO BAR | BAZ QUX" in out, f"Got: {out}"

    def test_regex_processing(self, nb_runner):
        """Test regex operations cached across cells."""
        nb_runner.create_notebook([
            "import re",
            "text = 'Hello 42 world 99 foo 7'",
            "numbers = re.findall(r'\\d+', text)\nprint(f'numbers = {numbers}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "numbers = ['42', '99', '7']" in out, f"Got: {out}"


class TestConditionalLogic:
    """Test conditional logic patterns."""

    def test_if_else_with_function(self, nb_runner):
        """Test if/else selecting different functions."""
        nb_runner.create_notebook([
            "mode = 'square'",
            """def square(x): return x ** 2
def double(x): return x * 2""",
            """if mode == 'square':
    func = square
else:
    func = double
result = func(5)
print(f'result = {result}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 25" in nb_runner.get_output(3)

    def test_change_condition_triggers_recompute(self, nb_runner):
        """Changing a condition variable should trigger recomputation."""
        nb_runner.create_notebook([
            "mode = 'add'",
            """if mode == 'add':
    result = 10 + 20
else:
    result = 10 * 20
print(f'result = {result}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 30" in nb_runner.get_output(2)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'multiply'")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "result = 200" in out, f"Expected result=200, got: {out}"


class TestDictComprehensionAndSet:
    """Test dict/set comprehension caching."""

    def test_dict_comprehension(self, nb_runner):
        """Dict comprehension should be cached correctly."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']\nvalues = [1, 2, 3]",
            "d = {k: v ** 2 for k, v in zip(keys, values)}\nprint(d)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "'a': 1" in out, f"Got: {out}"
        assert "'b': 4" in out
        assert "'c': 9" in out

    def test_set_operations_across_cells(self, nb_runner):
        """Test set operations across multiple cells."""
        nb_runner.create_notebook([
            "s1 = {1, 2, 3, 4, 5}",
            "s2 = {3, 4, 5, 6, 7}",
            "intersection = s1 & s2\nunion_set = s1 | s2\ndiff = s1 - s2",
            "print(f'inter={sorted(intersection)}, union={sorted(union_set)}, diff={sorted(diff)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(4)
        assert "inter=[3, 4, 5]" in out, f"Got: {out}"
        assert "union=[1, 2, 3, 4, 5, 6, 7]" in out
        assert "diff=[1, 2]" in out
