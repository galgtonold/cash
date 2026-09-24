"""Loops, conditionals and generators whose inputs come from cells above."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestNestedLoopPatterns:
    """Test nested loop patterns and their interaction with caching."""

    @pytest.mark.loops
    def test_while_loop_convergence(self, nb_runner):
        """While loop that converges to a value."""
        nb_runner.create_notebook(
            [
                "target = 100\ntolerance = 0.01",
                textwrap.dedent("""\
                value = 1.0
                iterations = 0
                while abs(value - target) > tolerance:
                    value = (value + target / value) / 2
                    iterations += 1
                    if iterations > 1000:
                        break"""),
                "print(f'Value: {value:.4f}, Iterations: {iterations}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Value: 10.0000" in out  # sqrt(100) = 10

    @pytest.mark.loops
    def test_loop_with_conditional_accumulation(self, nb_runner):
        """Loop that conditionally adds to different accumulators."""
        nb_runner.create_notebook(
            [
                "data = list(range(20))",
                textwrap.dedent("""\
                evens = []
                odds = []
                for x in data:
                    if x % 2 == 0:
                        evens.append(x)
                    else:
                        odds.append(x)"""),
                "print(f'Evens: {len(evens)}, Odds: {len(odds)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Evens: 10, Odds: 10" in out


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexLoopPatterns:
    """Tests for complex loop patterns that interact with caching."""

    @pytest.mark.loops
    def test_nested_loop_with_accumulator(self, nb_runner):
        """Nested loops with accumulator pattern."""
        nb_runner.create_notebook(
            [
                "n = 5",
                "matrix = []\nfor i in range(n):\n    row = []\n    for j in range(n):\n        row.append(i * n + j)\n    matrix.append(row)",
                "flat = [x for row in matrix for x in row]",
                "total = sum(flat)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "total=300" in output  # sum(0..24) = 300

        # Change n
        nb_runner.set_cell_source(1, "n = 3")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "total=36" in output2  # sum(0..8) = 36

    @pytest.mark.loops
    def test_while_loop_with_convergence(self, nb_runner):
        """While loop that converges to a result."""
        nb_runner.create_notebook(
            [
                "target = 100\ntolerance = 0.01",
                "guess = 1.0\niterations = 0\nwhile abs(guess * guess - target) > tolerance:\n    guess = (guess + target / guess) / 2\n    iterations += 1",
                "print(f'sqrt={guess:.4f} iterations={iterations}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "sqrt=10.0" in output

        # Change target
        nb_runner.set_cell_source(1, "target = 225\ntolerance = 0.01")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "sqrt=15.0" in output2


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestFunctionPlusLoopEdit:
    """Function used inside a loop, both edited."""

    def test_edit_function_used_in_loop(self, nb_runner):
        """Function called inside loop, redefine function."""
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "results = []\nfor i in range(4):\n    results.append(transform(i))",
                "print(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 2, 4, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "results = [0, 1, 4, 9]" in nb_runner.get_output(3)

    def test_edit_loop_and_function(self, nb_runner):
        """Edit both the loop and the function it uses."""
        nb_runner.create_notebook(
            [
                "def scale(x, factor):\n    return x * factor",
                "out = []\nfor v in [1, 2, 3]:\n    out.append(scale(v, 10))",
                "print(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = [10, 20, 30]" in nb_runner.get_output(3)

        # Edit function and loop data
        nb_runner.set_cell_source(1, "def scale(x, factor):\n    return x + factor")
        nb_runner.set_cell_source(2, "out = []\nfor v in [10, 20, 30]:\n    out.append(scale(v, 100))")
        nb_runner.run_all()
        assert "out = [110, 120, 130]" in nb_runner.get_output(3)


@pytest.mark.core
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


@pytest.mark.core
class TestConditionalExecution:
    """Test conditional patterns across cells."""

    def test_conditional_variable_assignment(self, nb_runner):
        """Condition in cell 1 affects cell 2's computation."""
        nb_runner.create_notebook(
            [
                "mode = 'double'",
                "x = 10\nresult = x * 2 if mode == 'double' else x * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'triple'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 30" in out, f"Expected 30, got: {out}"


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestConditionalChainEdits:
    """Conditional logic chains with edits."""

    def test_nested_conditionals_edit(self, nb_runner):
        """Nested if-else, edit the input to change branch taken."""
        nb_runner.create_notebook(
            [
                "score = 85",
                "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelse:\n    grade = 'C'",
                "print(f'grade = {grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "score = 50")
        nb_runner.run_all()
        assert "grade = C" in nb_runner.get_output(3)

    def test_conditional_chain_edit_thresholds(self, nb_runner):
        """Edit the thresholds in conditionals."""
        nb_runner.create_notebook(
            [
                "val = 50",
                "if val > 100:\n    cat = 'high'\nelse:\n    cat = 'low'",
                "print(f'cat = {cat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cat = low" in nb_runner.get_output(3)

        # Lower threshold
        nb_runner.set_cell_source(2, "if val > 40:\n    cat = 'high'\nelse:\n    cat = 'low'")
        nb_runner.run_all()
        assert "cat = high" in nb_runner.get_output(3)


@pytest.mark.core
class TestGeneratorAndIteratorCaching:
    """Test caching behavior with generators and iterators."""

    def test_list_comprehension_from_range(self, nb_runner):
        """List comprehension result should be cached."""
        nb_runner.create_notebook(
            [
                "n = 5",
                "squares = [x**2 for x in range(n)]\nprint(squares)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run — should use cache
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)
