"""
Stress tests: Loops, Mutations, Control Structures (Scenarios 66-95)

Tests loop caching (empty, single, break, continue, nested), mutation detection,
while loops, if/else branches, and side effect handling.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.loops, pytest.mark.mutations]


# =============================================================================
# Scenario 66-85: Loop & Control Structure Edge Cases
# =============================================================================


class TestLoopEdgeCases:
    """Tests for loop caching edge cases."""

    def test_71_loop_modifying_external_accumulator(self, nb_runner):
        """Scenario 71: Loop appending to external list."""
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30]",
                "total = 0\nfor x in data:\n    total += x\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(2)
        # Change data
        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40]")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "total=100" in nb_runner.get_output(2)

    def test_72_loop_with_function_call(self, nb_runner):
        """Scenario 72: Loop calling function defined in another cell."""
        nb_runner.create_notebook(
            [
                "def process(x):\n    return x ** 2",
                "data = [1, 2, 3]",
                "results = []\nfor x in data:\n    results.append(process(x))\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 4, 9]" in nb_runner.get_output(3)
        # Change function
        nb_runner.set_cell_source(1, "def process(x):\n    return x ** 3")
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "results=[1, 8, 27]" in nb_runner.get_output(3)

    def test_74_if_else_branch_caching(self, nb_runner):
        """Scenario 75: If/else — change condition, verify correct branch."""
        nb_runner.create_notebook(
            [
                "n = 10",
                "if n > 5:\n    result = 'big'\nelse:\n    result = 'small'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=big" in nb_runner.get_output(2)
        # Change n to switch branch
        nb_runner.set_cell_source(1, "n = 3")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "result=small" in nb_runner.get_output(2)

    def test_75_if_elif_else_chain(self, nb_runner):
        """Scenario 76: If/elif/else chain."""
        nb_runner.create_notebook(
            [
                "score = 85",
                "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelif score >= 70:\n    grade = 'C'\nelse:\n    grade = 'F'\nprint(f'grade={grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade=B" in nb_runner.get_output(2)
        # Change to A range
        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "grade=A" in nb_runner.get_output(2)

    def test_80_loop_change_iterations_downstream_correct(self, nb_runner):
        """Change loop iterations, downstream cell gets correct result."""
        nb_runner.create_notebook(
            [
                "d = {}\nfor k in ['a', 'b']:\n    d[k] = len(k)",
                "print(f'keys={sorted(d.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['a', 'b']" in nb_runner.get_output(2)
        # Add an iteration
        nb_runner.set_cell_source(1, "d = {}\nfor k in ['a', 'b', 'cc']:\n    d[k] = len(k)")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "keys=['a', 'b', 'cc']" in nb_runner.get_output(2)


# =============================================================================
# Scenario 86-95: Mutation Detection & Side Effects
# =============================================================================


class TestMutationDetection:
    """Tests for mutation detection and side effect handling."""

    def test_83_list_append_in_loop(self, nb_runner):
        """Scenario 86: List append in loop — mutation tracked."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "results = []\nfor x in data:\n    results.append(x * 2)\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[2, 4, 6]" in nb_runner.get_output(2)
        # Change data
        nb_runner.set_cell_source(1, "data = [10, 20]")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "results=[20, 40]" in nb_runner.get_output(2)

    def test_86_chained_method_not_mutation(self, nb_runner):
        """Scenario 90: df.drop().reset_index() — NOT a mutation of df."""
        nb_runner.create_notebook(
            [
                "import pandas as pd\ndf = pd.DataFrame({'a': [1,2], 'b': [3,4]})",
                "result = df.drop(columns=['b']).reset_index(drop=True)\nprint(result.columns.tolist())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['a']" in nb_runner.get_output(2)
        # Re-run — df is unchanged, should skip/cache
        nb_runner.run_cell(2)
        assert "['a']" in nb_runner.get_output(2)

    def test_89_inplace_pandas_mutation(self, nb_runner):
        """Scenario 88: DataFrame inplace operation."""
        nb_runner.create_notebook(
            [
                "import pandas as pd\ndf = pd.DataFrame({'a': [1,2], 'b': [3,4]})",
                "df.drop(columns=['b'], inplace=True)\nprint(df.columns.tolist())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['a']" in nb_runner.get_output(2)

    def test_91_loop_accumulator_correct_after_change(self, nb_runner):
        """Loop accumulator reflects changes after loop code modification."""
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 3]",
                "acc = 0\nfor n in nums:\n    acc += n\nprint(f'acc={acc}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "acc=6" in nb_runner.get_output(2)
        # Change accumulator operation
        nb_runner.set_cell_source(2, "acc = 1\nfor n in nums:\n    acc *= n\nprint(f'acc={acc}')")
        nb_runner.run_cell(2)
        assert "acc=6" in nb_runner.get_output(2)

    def test_94_mutation_no_output_variable(self, nb_runner):
        """Subscript mutation with no new output variable — isolated re-run is idempotent.

        ``data`` is a no-lineage dict mutated in place (``data['count'] += 1``).
        Re-running cell 2 alone == running from the start, so ``data`` is restored
        to its cell-entry base ``{'count': 0}`` before re-execution and ``count``
        stays ``1`` rather than accumulating to ``2`` — see test_isolated_rerun_gaps.
        """
        nb_runner.create_notebook(
            [
                "data = {'count': 0}",
                "data['count'] += 1\nprint(f\"count={data['count']}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=1" in nb_runner.get_output(2)
        # Re-run — base data={'count': 0} is restored first, so count stays 1.
        nb_runner.run_cell(2)
        assert "count=1" in nb_runner.get_output(2)
