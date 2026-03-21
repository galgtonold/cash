"""
Test: _check_lineage_based should skip variables produced by control structures.

Reproduces the bug where:
1. A for loop runs as a single unit (fast-loop optimization)
2. Variables produced by the for loop get their lineage set by the
   ControlStructureProcessor (value-based hashing)
3. On re-execution of a downstream cell, _check_lineage_based tries to verify
   the for-loop-produced variable's lineage using a DIFFERENT formula
   (simple input-lineage composition)
4. The formula mismatch always produces a false positive, triggering
   re-execution of the entire for loop

The fix: _check_lineage_based skips variables whose executed_cell_codes
points to a control structure (for/while/if/with/try).
"""

import time
import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(60)]


class TestLineageCheckControlStructure:
    """Verify _check_lineage_based doesn't re-execute control structures."""

    def test_second_run_skips_for_loop_reexecution(self, nb_runner):
        """
        When a cell is executed a second time, the upstream check should NOT
        re-execute a for loop that hasn't changed. Previously, the lineage
        formula mismatch caused the for loop to re-execute every time.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np\nimport time",
            # Cell 2: cash_on + debug
            "%cash_on\n%cash_debug on",
            # Cell 3: setup
            (
                "N = 10\n"
                "data = np.zeros(N)"
            ),
            # Cell 4: for loop that modifies data (will run as single unit)
            (
                "result = np.zeros(N)\n"
                "accumulator = 0.0\n"
                "for i in range(1000):\n"
                "    result = result + np.random.randn(N) * 0.001\n"
                "    accumulator += np.sum(result)\n"
                "p = result.sum()\n"
                "print(f'Loop done: p={p:.4f}')"
            ),
            # Cell 5: downstream cell that uses p
            "print(f'p = {p:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output5 = nb_runner.get_output(5)
        assert "p =" in output5

        # Run cell 5 a second time — this should be fast
        t0 = time.time()
        nb_runner.run_cell(5)
        elapsed = time.time() - t0

        output5_2 = nb_runner.get_output(5)
        assert "p =" in output5_2

        # The second run should be very fast (< 5s).
        # Without the fix, it would re-execute the 1000-iteration loop
        # during _check_lineage_based, taking much longer.
        assert elapsed < 5.0, (
            f"Second run took {elapsed:.1f}s — likely re-executing the for loop "
            f"due to lineage formula mismatch in _check_lineage_based"
        )

    def test_second_run_skips_for_loop_cfd_pattern(self, nb_runner):
        """
        Mimics the exact CFD pattern: a for loop produces variables like p, u, v,
        and a downstream cell saves them. Second execution should be instant.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np\nimport time",
            # Cell 2: cash_on
            "%cash_on",
            # Cell 3: setup
            "N = 10\ndx = 0.1",
            # Cell 4: for loop (runs as single unit due to many iterations)
            (
                "u = np.zeros(N)\n"
                "v = np.zeros(N)\n"
                "p = np.zeros(N)\n"
                "residual_history = []\n"
                "for step in range(5000):\n"
                "    u = u + 0.001 * np.sin(np.linspace(0, np.pi, N))\n"
                "    v = v + 0.0005 * np.cos(np.linspace(0, np.pi, N))\n"
                "    p = u + v\n"
                "    residual_history.append(np.max(np.abs(p)))\n"
                "print(f'Done: {len(residual_history)} steps')"
            ),
            # Cell 5: post-processing with intermediates
            (
                "vorticity = (v[2:] - v[:-2]) / (2 * dx)\n"
                "stream = np.cumsum(vorticity)\n"
                "print(f'Vorticity range: [{vorticity.min():.4f}, {vorticity.max():.4f}]')"
            ),
            # Cell 6: downstream using stream and u
            "print(f'stream sum: {stream.sum():.4f}, u max: {u.max():.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output6 = nb_runner.get_output(6)
        assert "stream sum:" in output6

        # Run cell 6 a second time
        t0 = time.time()
        nb_runner.run_cell(6)
        elapsed = time.time() - t0

        output6_2 = nb_runner.get_output(6)
        assert "stream sum:" in output6_2

        # Should be fast — no for loop re-execution
        assert elapsed < 5.0, (
            f"Second run took {elapsed:.1f}s — for loop was likely re-executed"
        )

    def test_for_loop_still_reexecutes_when_inputs_change(self, nb_runner):
        """
        Ensure the fix doesn't break legitimate re-execution: when an upstream
        variable changes, the for loop SHOULD be scheduled for re-execution
        by the notebook-based checker (not the lineage-based one).
        """
        nb_runner.create_notebook([
            "import numpy as np",
            "%cash_on",
            "N = 10",
            (
                "result = np.zeros(N)\n"
                "for i in range(100):\n"
                "    result = result + 1\n"
                "total = result.sum()\n"
                "print(f'Total: {total}')"
            ),
            "print(f'Total is: {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output5 = nb_runner.get_output(5)
        assert "Total is: 1000" in output5  # 10 elements * 100 iterations

        # Change N → for loop should be invalidated by notebook-based checker
        nb_runner.set_cell_source(3, "N = 20")
        nb_runner.run_cells([3, 5])

        output5_mod = nb_runner.get_output(5)
        assert "Total is: 2000" in output5_mod  # 20 elements * 100 iterations
