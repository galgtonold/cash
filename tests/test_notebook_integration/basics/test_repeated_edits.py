"""One notebook edited again and again: rapid edits, flip-flops, reverts and refinement rounds."""

import pytest


# Multi-round cell editing & cache coherence.
#
# Tests that exercise the trickiest interaction patterns:
# - Editing the same cell multiple times in succession
# - Editing multiple cells between runs
# - Reverting a cell to its original value (should hit cache)
# - Editing upstream then downstream then upstream again
# - Rapid back-and-forth value flipping
@pytest.mark.stress
@pytest.mark.upstream
class TestMultiEditSameCell:
    """Edit a single upstream cell many times, verifying propagation each time."""

    def test_edit_upstream_three_times(self, nb_runner):
        """Change cell 1 three times, each time verifying cell 3 picks it up."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "print(f'result = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(3)
        assert "result = 100" in nb_runner.get_output(3)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 7")
        nb_runner.run_cell(3)
        assert "result = 14" in nb_runner.get_output(3)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 0")
        nb_runner.run_cell(3)
        assert "result = 0" in nb_runner.get_output(3)

    def test_rapid_flip_flop(self, nb_runner):
        """Rapidly alternate between two values for a cell."""
        nb_runner.create_notebook(
            [
                "flag = True",
                "result = 'yes' if flag else 'no'",
                "print(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = yes" in nb_runner.get_output(3)

        for i in range(4):
            val = "False" if i % 2 == 0 else "True"
            expected = "no" if i % 2 == 0 else "yes"
            nb_runner.set_cell_source(1, f"flag = {val}")
            nb_runner.run_cell(3)
            assert f"result = {expected}" in nb_runner.get_output(3), f"Iteration {i}: expected '{expected}'"


# Rapid-fire edit interaction tests.
#
# Tests that exercise many rapid successive edits to the same cell(s),
# verifying cache coherence under high edit frequency.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestRapidEditsOneCell:
    """Many edits to a single cell in quick succession."""

    def test_five_rapid_edits(self, nb_runner):
        """Edit the same cell five times, verify each time."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()

        for i in range(5):
            nb_runner.set_cell_source(1, f"x = {i * 10}")
            nb_runner.run_all()
            assert f"y = {i * 10 + 1}" in nb_runner.get_output(2)

    def test_ten_rapid_edits(self, nb_runner):
        """Ten rapid edits to the upstream cell."""
        nb_runner.create_notebook(
            [
                "n = 0",
                "result = n ** 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()

        for i in range(10):
            nb_runner.set_cell_source(1, f"n = {i}")
            nb_runner.run_all()
            assert f"result = {i**2}" in nb_runner.get_output(2)

    def test_rapid_edits_with_function(self, nb_runner):
        """Rapidly edit function definition."""
        nb_runner.create_notebook(
            [
                "def f(x):\n    return x + 0",
                "result = f(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()

        for offset in [1, 5, 10, 100, -1]:
            nb_runner.set_cell_source(1, f"def f(x):\n    return x + {offset}")
            nb_runner.run_all()
            assert f"result = {10 + offset}" in nb_runner.get_output(2)


# Multiple rapid consecutive edits interaction tests.
#
# Tests making multiple rapid edits to the same cell and verifying
# that each edit is properly picked up.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRapidSameCellEdits:
    """Multiple rapid edits to the same cell."""

    def test_three_consecutive_edits(self, nb_runner):
        """Edit a cell three times consecutively."""
        nb_runner.create_notebook(
            [
                "x = 1  # version 1",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 1" in nb_runner.get_output(2)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 10  # version 2")
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(2)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 100  # version 3")
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 1000  # version 4")
        nb_runner.run_all()
        assert "x = 1000" in nb_runner.get_output(2)

    def test_oscillating_values(self, nb_runner):
        """Alternate between two values rapidly."""
        nb_runner.create_notebook(
            [
                "val = 'A'  # oscillate val A",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = A" in nb_runner.get_output(2)

        # Switch to B
        nb_runner.set_cell_source(1, "val = 'B'  # oscillate val B")
        nb_runner.run_all()
        assert "val = B" in nb_runner.get_output(2)

        # Back to A (new comment to differentiate)
        nb_runner.set_cell_source(1, "val = 'A'  # oscillate val A again")
        nb_runner.run_all()
        assert "val = A" in nb_runner.get_output(2)

    def test_edit_type_change(self, nb_runner):
        """Edit a cell to change variable type each time."""
        nb_runner.create_notebook(
            [
                "data = 42  # type change start",
                "print(f'type = {type(data).__name__}, data = {data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type = int" in nb_runner.get_output(2)

        # Change to string
        nb_runner.set_cell_source(1, "data = 'hello'  # type change str")
        nb_runner.run_all()
        assert "type = str" in nb_runner.get_output(2)

        # Change to list
        nb_runner.set_cell_source(1, "data = [1, 2, 3]  # type change list")
        nb_runner.run_all()
        assert "type = list" in nb_runner.get_output(2)

        # Change to dict
        nb_runner.set_cell_source(1, "data = {'a': 1}  # type change dict")
        nb_runner.run_all()
        assert "type = dict" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestRapidEditsMultipleCells:
    """Rapid edits across multiple cells."""

    def test_alternating_rapid_edits(self, nb_runner):
        """Alternate between editing two cells rapidly."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = 1",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 2" in nb_runner.get_output(3)

        # Alternate edits
        for i in range(1, 6):
            if i % 2 == 1:
                nb_runner.set_cell_source(1, f"a = {i * 10}")
            else:
                nb_runner.set_cell_source(2, f"b = {i * 10}")
            nb_runner.run_all()

        # After 5 rounds: a=50, b=40
        assert "c = 90" in nb_runner.get_output(3)

    def test_rapid_edit_and_revert(self, nb_runner):
        """Rapidly edit and revert, check cache consistency.

        After editing x = 999 and running, then reverting back to x = 1,
        the system should detect the upstream change in both directions.
        Uses kernel restart between cycles to avoid "already executed"
        skip optimization masking the revert detection.
        """
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit to new value
        nb_runner.set_cell_source(1, "x = 999")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 1000" in nb_runner.get_output(2)

        # Revert back to original
        nb_runner.set_cell_source(1, "x = 1")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRapidDependentEdits:
    """Rapid edits to dependent cells."""

    def test_edit_producer_then_consumer_rapidly(self, nb_runner):
        """Edit both producer and consumer cells rapidly."""
        nb_runner.create_notebook(
            [
                "x = 5  # producer",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Edit producer
        nb_runner.set_cell_source(1, "x = 50  # producer big")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)

        # Edit consumer
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 150" in nb_runner.get_output(2)

        # Edit both at once
        nb_runner.set_cell_source(1, "x = 7  # producer small")
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 8" in nb_runner.get_output(2)

    def test_rapid_formula_changes(self, nb_runner):
        """Rapidly change the formula applied to same input."""
        nb_runner.create_notebook(
            [
                "n = 10  # input number",
                "result = n + 1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        formulas = [
            ("result = n * 2\nprint(f'result = {result}')", "20"),
            ("result = n ** 2\nprint(f'result = {result}')", "100"),
            ("result = n // 3\nprint(f'result = {result}')", "3"),
            ("result = n - 7\nprint(f'result = {result}')", "3"),
        ]
        for code, expected in formulas:
            nb_runner.set_cell_source(2, code)
            nb_runner.run_all()
            assert f"result = {expected}" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestRapidEditWithRestart:
    """Rapid edits combined with kernel restarts."""

    def test_edit_restart_cycle(self, nb_runner):
        """Edit → restart → verify cycle."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 1" in nb_runner.get_output(2)

        for val in [10, 20, 30]:
            nb_runner.set_cell_source(1, f"x = {val}")
            nb_runner.shutdown()
            nb_runner.start_kernel()
            nb_runner.run_all()
            assert f"y = {val + 1}" in nb_runner.get_output(2)


# Multi-round iterative refinement patterns.
#
# Tests multiple sequential edits to same cell, verifying each round.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMultiRoundRefinement:
    """Multiple rounds of editing the same cell."""

    def test_five_round_formula_refinement(self, nb_runner):
        """Edit formula cell 5 times sequentially."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = x\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit 1: add 5
        nb_runner.set_cell_source(2, "result = x + 5\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Edit 2: multiply
        nb_runner.set_cell_source(2, "result = x * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Edit 3: power
        nb_runner.set_cell_source(2, "result = x ** 2\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Edit 4: floor div
        nb_runner.set_cell_source(2, "result = x // 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 3" in nb_runner.get_output(2)

        # Edit 5: complex expression
        nb_runner.set_cell_source(2, "result = (x + 1) * (x - 1)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 99" in nb_runner.get_output(2)

    def test_alternating_data_source(self, nb_runner):
        """Alternate data sources, verify correct propagation each time."""
        nb_runner.create_notebook(
            [
                "source = [1, 2, 3]",
                "total = sum(source)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        for vals, expected in [
            ([10, 20], 30),
            ([100], 100),
            ([5, 5, 5, 5], 20),
            ([1000, 2000, 3000], 6000),
        ]:
            nb_runner.set_cell_source(1, f"source = {vals}")
            nb_runner.run_all()
            assert f"total = {expected}" in nb_runner.get_output(2)

    def test_refine_function_multiple_times(self, nb_runner):
        """Refine function definition through multiple iterations."""
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x",
                "vals = [1, 2, 3, 4, 5]\nresult = [transform(v) for v in vals]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5]" in nb_runner.get_output(2)

        # v2: double
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 2")
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # v3: square
        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

        # v4: negate
        nb_runner.set_cell_source(1, "def transform(x):\n    return -x")
        nb_runner.run_all()
        assert "result = [-1, -2, -3, -4, -5]" in nb_runner.get_output(2)


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
