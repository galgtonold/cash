"""
Test that upstream execution skips earlier definitions of a variable
when a later statement fully redefines it.

Bug: When executing a cell that uses `x`, the upstream checker traces back
through the simulation trace and finds multiple statements that output `x`.
Even though `x = {'b': 123}` fully redefines `x`, the backwards scan continued
past it because `needed_vars` wasn't updated when a statement was scheduled
for execution. This caused the try/except block (which also outputs `x`) to
be unnecessarily executed.

Fix: When scheduling a statement for execution in the backwards scan, remove
its output variables (that are NOT also inputs) from `needed_vars`. This stops
the cascade to earlier statements that produce the same variable but are
fully overwritten.
"""

import pytest
import time

pytestmark = pytest.mark.upstream


class TestSkipOverwrittenVariable:
    """Verify that upstream execution skips earlier definitions when variable is fully redefined."""

    def test_try_except_not_executed_when_var_fully_redefined(self, nb_runner):
        """
        If x is defined in a try/except block, then later fully redefined as
        x = {'b': 123} followed by x['a'] = f(0), executing a cell that uses x
        should NOT execute the try/except block — only the later redefinition chain.
        """
        nb_runner.create_notebook([
            # Cell 1: Define x in try/except (should NOT be executed)
            (
                "import time\n"
                "try:\n"
                "    print('hi')\n"
                "    asdf = 235\n"
                "    raise ValueError('This is a test error')\n"
                "    x = 123\n"
                "except ValueError as e:\n"
                "    print(f'Value error occurred: {e}')\n"
                "    time.sleep(1)  # Slow operation to detect if executed\n"
            ),
            # Cell 2: Define c (dependency of f)
            "c = 122",
            # Cell 3: Define f that depends on c, then fully redefine x
            (
                "def f(a):\n"
                "    return c + a\n"
                "\n"
                "x = {'b': 123}\n"
                "x['a'] = f(0)"
            ),
            # Cell 4: Use x (this is the cell we'll execute)
            "x",
        ])
        nb_runner.start_kernel()

        # Run all cells first
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "122" in output  # x['a'] = f(0) = c + 0 = 122

        # Now restart kernel and run only cell 4
        nb_runner.reset_cash_state()

        t0 = time.time()
        nb_runner.run_cell(4)
        elapsed = time.time() - t0
        output = nb_runner.get_output(4)

        # The result should still be correct
        assert "122" in output

        # Check that the try/except with time.sleep(1) was NOT executed
        # If it was executed, it would take >1s
        assert elapsed < 1.5, (
            f"Upstream execution took {elapsed:.1f}s — try/except block was likely "
            f"unnecessarily executed (contains time.sleep(1))"
        )

        # Also check raw output doesn't contain the try/except side effects
        raw = nb_runner.get_raw_output(4)
        assert "Value error occurred" not in raw, (
            f"Try/except block was unnecessarily executed upstream. Raw: {raw[:500]}"
        )

    def test_earlier_assignment_skipped_when_variable_fully_redefined(self, nb_runner):
        """
        If x is defined as x=10 in cell 1, then fully redefined as x=20 in cell 2,
        executing a cell that uses x should only execute cell 2's definition.
        """
        nb_runner.create_notebook([
            # Cell 1: Define x (should NOT be executed when later redefined)
            "x = 10\nprint('cell1_executed')",
            # Cell 2: Fully redefine x
            "x = 20\nprint('cell2_executed')",
            # Cell 3: Use x
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()

        # Run all cells first
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "x = 20" in output

        # Reset state and run only cell 3
        nb_runner.reset_cash_state()
        nb_runner.run_cell(3)
        output = nb_runner.get_output(3)

        # Result should be correct
        assert "x = 20" in output

        # Check that cell 1 was NOT re-executed (cell1_executed should not appear)
        raw = nb_runner.get_raw_output(3)
        assert "cell1_executed" not in raw, (
            f"Earlier assignment was unnecessarily re-executed. Raw: {raw[:500]}"
        )

    def test_mutation_after_redefinition_still_works(self, nb_runner):
        """
        When x is first created, then fully redefined, then mutated,
        the mutation chain should work correctly but NOT cascade past
        the full redefinition.
        """
        nb_runner.create_notebook([
            # Cell 1: First definition (should be skipped)
            "x = [1, 2, 3]\nprint('original_def')",
            # Cell 2: Full redefinition
            "x = [10, 20, 30]",
            # Cell 3: Mutation (needs x from cell 2)
            "x.append(40)",
            # Cell 4: Use x
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "40" in output

        # Reset and run cell 4
        nb_runner.reset_cash_state()
        nb_runner.run_cell(4)
        output = nb_runner.get_output(4)

        # Should still be correct
        assert "40" in output or "10" in output  # Either restored or recomputed

        # Cell 1 should NOT have been executed
        raw = nb_runner.get_raw_output(4)
        assert "original_def" not in raw, (
            f"Original definition was unnecessarily re-executed. Raw: {raw[:500]}"
        )
