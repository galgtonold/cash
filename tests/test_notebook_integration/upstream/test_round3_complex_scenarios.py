"""
Complex scenario integration tests.

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

    def test_six_cell_chain_modification_propagates(self, nb_runner):
        """
        Run a 6-cell chain, modify cell 1, then re-run cell 6.
        The upstream simulation should detect the change and recompute.
        """
        nb_runner.create_notebook(
            [
                "base = 10",
                "step1 = base * 2",
                "step2 = step1 + 5",
                "step3 = step2 ** 2",
                "step4 = step3 - 100",
                "print(f'Result: {step4}')",
            ]
        )
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

    def test_branching_modify_root_all_branches_update(self, nb_runner):
        """
        Modify the root of a diamond and verify both branches update.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x * 2",
                "b = x + 5",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
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

    def test_dataclass_mutation_detection(self, nb_runner):
        """Test that mutating a dataclass field is detected."""
        nb_runner.create_notebook(
            [
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
            ]
        )
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
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "result = transform(5)\nprint(f'Result: {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "def compute(x):\n    return x + 100",
                "val = compute(5)\nprint(f'val = {val}')",
            ]
        )
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

    def test_nested_tuple_unpacking(self, nb_runner):
        """Test nested tuple unpacking."""
        nb_runner.create_notebook(
            [
                "pairs = [(1, 'a'), (2, 'b'), (3, 'c')]",
                "firsts = [x for x, y in pairs]\nseconds = [y for x, y in pairs]",
                "print(f'firsts={firsts}, seconds={seconds}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "firsts=[1, 2, 3]" in out, f"Got: {out}"
        assert "seconds=['a', 'b', 'c']" in out


class TestCrossDataTransformations:
    """Test complex data transformations across cells."""

    def test_pandas_modify_source_data(self, nb_runner):
        """Modify source data and verify the pipeline updates."""
        nb_runner.create_notebook(
            [
                """import pandas as pd
df = pd.DataFrame({'name': ['A', 'B'], 'val': [10, 20]})""",
                "total = df['val'].sum()\nprint(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Total: 30" in nb_runner.get_output(2)

        # Modify the DataFrame
        nb_runner.set_cell_source(
            1, "import pandas as pd\ndf = pd.DataFrame({'name': ['A', 'B', 'C'], 'val': [10, 20, 30]})"
        )
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Total: 60" in out, f"Expected Total: 60, got: {out}"


class TestReexecutionPatterns:
    """Test various re-execution patterns."""

    def test_skip_optimization_on_rerun(self, nb_runner):
        """
        Run all cells, then re-run them all.
        Second run should skip (or cache-hit) unchanged cells.
        """
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x + 8\nprint(f'y = {y}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 99",
                "result = a * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 20" in nb_runner.get_output(3)

        # Now make result depend on b too
        nb_runner.set_cell_source(3, "result = a + b\nprint(f'result = {result}')")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "result = 109" in out, f"Expected result=109, got: {out}"


class TestClassInheritancePatterns:
    """Test class inheritance and method caching."""

    def test_modify_base_class_invalidates_child(self, nb_runner):
        """Modifying base class should invalidate derived class usage."""
        nb_runner.create_notebook(
            [
                """class Base:
    def greet(self):
        return "Hello"
""",
                """class Child(Base):
    def greet(self):
        return super().greet() + " World"
""",
                "obj = Child()\nprint(obj.greet())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Hello World" in nb_runner.get_output(3)

        # Modify the base class
        nb_runner.set_cell_source(
            1,
            """class Base:
    def greet(self):
        return "Hi"
""",
        )
        nb_runner.run_cells([1, 2, 3])

        out = nb_runner.get_output(3)
        assert "Hi World" in out, f"Expected 'Hi World', got: {out}"


class TestMultiStatementCells:
    """Test cells with many statements."""

    def test_mixed_assignments_and_expressions(self, nb_runner):
        """Test cells mixing assignments, print calls, and function definitions."""
        nb_runner.create_notebook(
            [
                """data = [1, 2, 3, 4, 5]
total = sum(data)
mean = total / len(data)
above = [x for x in data if x > mean]
below = [x for x in data if x <= mean]
print(f'mean={mean}, above={above}, below={below}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "mean=3.0" in out, f"Got: {out}"
        assert "above=[4, 5]" in out
        assert "below=[1, 2, 3]" in out


class TestConditionalLogic:
    """Test conditional logic patterns."""

    def test_change_condition_triggers_recompute(self, nb_runner):
        """Changing a condition variable should trigger recomputation."""
        nb_runner.create_notebook(
            [
                "mode = 'add'",
                """if mode == 'add':
    result = 10 + 20
else:
    result = 10 * 20
print(f'result = {result}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 30" in nb_runner.get_output(2)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'multiply'")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "result = 200" in out, f"Expected result=200, got: {out}"
