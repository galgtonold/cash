"""
Test: upstream checker must detect when for-loop CODE changes and re-execute dependents.

Reproduces scenarios where:
1. All cells execute (including for-loop and post-processing)
2. The for-loop cell code is modified
3. A downstream cell is re-run
4. The upstream checker should detect the change and re-execute dependents

Also tests the "unsaved edit" scenario: user edits and executes a cell
in VS Code but the .ipynb file on disk hasn't been saved yet.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.loops, pytest.mark.timeout(60)]


class TestLoopCodeChangeDetection:
    """Verify that modifying for-loop code triggers re-execution of dependents."""

    def test_change_loop_body_detected_by_downstream(self, nb_runner):
        """
        Modify the for-loop body and verify the downstream cell gets
        updated results.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np",
            # Cell 2: cash_on + debug
            "%cash_on\n%cash_debug on",
            # Cell 3: setup
            "N = 10\ndx = 0.1",
            # Cell 4: for loop
            (
                "u = np.zeros(N)\n"
                "v = np.zeros(N)\n"
                "p = np.zeros(N)\n"
                "for step in range(100):\n"
                "    u = u + 0.01\n"
                "    v = v + 0.005\n"
                "    p = u + v\n"
                "print(f'u_max={u.max():.4f}, v_max={v.max():.4f}')"
            ),
            # Cell 5: post-processing that depends on u, v from loop
            (
                "vorticity = (v[2:] - v[:-2]) / (2 * dx)\n"
                "stream = np.cumsum(u)\n"
                "print(f'stream_sum={stream.sum():.4f}')"
            ),
            # Cell 6: downstream using stream
            "result = stream.sum()\nprint(f'result={result:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Verify initial values
        output6 = nb_runner.get_output(6)
        assert "result=" in output6

        # Now modify the for-loop: add an extra operation
        nb_runner.set_cell_source(4, (
            "u = np.zeros(N)\n"
            "v = np.zeros(N)\n"
            "p = np.zeros(N)\n"
            "for step in range(100):\n"
            "    u = u + 0.01\n"
            "    v = v + 0.005\n"
            "    p = u + v\n"
            "    u = u + 1\n"  # <-- NEW LINE: adds 1 to u each iteration
            "print(f'u_max={u.max():.4f}, v_max={v.max():.4f}')"
        ))

        # Run cell 6 — should detect loop change and re-execute chain
        nb_runner.run_cell(6)

        output6_mod = nb_runner.get_output(6)
        assert "result=" in output6_mod

        # The old result was based on u.max() = 1.0 (100 * 0.01)
        # The new result should reflect u.max() = 101.0 (100 * 0.01 + 100 * 1.0)
        # stream = cumsum(u) where u is much larger now
        # Extract the number and verify it changed
        import re
        old_match = re.search(r'result=([\d.]+)', output6.strip())
        new_match = re.search(r'result=([\d.]+)', output6_mod.strip())
        assert old_match and new_match, f"Could not parse output: old={output6}, new={output6_mod}"

        old_val = float(old_match.group(1))
        new_val = float(new_match.group(1))
        assert new_val > old_val * 10, (
            f"Loop code change not detected! old={old_val}, new={new_val}. "
            f"Expected new value to be much larger since u += 1 was added."
        )

    def test_change_loop_body_intermediate_cell_recomputes(self, nb_runner):
        """
        When the for-loop changes, intermediate cells (between loop and
        downstream) should also be re-executed with the new values.
        """
        nb_runner.create_notebook([
            "import numpy as np",
            "%cash_on\n%cash_debug on",
            "N = 5",
            # Cell 4: simple for loop
            (
                "data = np.zeros(N)\n"
                "for i in range(50):\n"
                "    data = data + 0.1\n"
                "total = data.sum()\n"
                "print(f'total={total:.4f}')"
            ),
            # Cell 5: depends on data from loop
            (
                "processed = data * 2\n"
                "print(f'processed_sum={processed.sum():.4f}')"
            ),
            # Cell 6: depends on processed
            "final = processed.sum()\nprint(f'final={final:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output6 = nb_runner.get_output(6)
        assert "final=" in output6
        # data.sum() = 5 * 50 * 0.1 = 25.0, processed = data * 2, sum = 50.0
        assert "50.0000" in output6

        # Modify the loop: double the increment
        nb_runner.set_cell_source(4, (
            "data = np.zeros(N)\n"
            "for i in range(50):\n"
            "    data = data + 0.2\n"  # Changed from 0.1 to 0.2
            "total = data.sum()\n"
            "print(f'total={total:.4f}')"
        ))

        # Run cell 6 — should detect loop change, re-execute loop + cell 5
        nb_runner.run_cell(6)

        output6_mod = nb_runner.get_output(6)
        assert "final=" in output6_mod
        # data.sum() = 5 * 50 * 0.2 = 50.0, processed = data * 2, sum = 100.0
        assert "100.0000" in output6_mod, (
            f"Loop code change not propagated to cell 6! Got: {output6_mod}"
        )

    def test_change_loop_body_after_second_run(self, nb_runner):
        """
        The user's exact scenario: 
        1. Run all cells → works
        2. Run downstream cell again → caches correctly (no re-execution)
        3. Modify the for-loop
        4. Run downstream cell → MUST detect change and re-execute
        """
        nb_runner.create_notebook([
            "import numpy as np",
            "%cash_on\n%cash_debug on",
            "N = 5",
            # Cell 4: for loop
            (
                "arr = np.zeros(N)\n"
                "for i in range(100):\n"
                "    arr = arr + 1\n"
                "total = arr.sum()\n"
                "print(f'total={total:.1f}')"
            ),
            # Cell 5: depends on arr
            "derived = arr.max()\nprint(f'derived={derived:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output5 = nb_runner.get_output(5)
        assert "derived=100.0" in output5

        # Run cell 5 a second time — should be cached
        nb_runner.run_cell(5)
        output5_cached = nb_runner.get_output(5)
        assert "derived=100.0" in output5_cached

        # NOW modify the loop
        nb_runner.set_cell_source(4, (
            "arr = np.zeros(N)\n"
            "for i in range(100):\n"
            "    arr = arr + 2\n"  # Changed from +1 to +2
            "total = arr.sum()\n"
            "print(f'total={total:.1f}')"
        ))

        # Run cell 5 — MUST detect loop change
        nb_runner.run_cell(5)
        output5_after_change = nb_runner.get_output(5)
        assert "derived=200.0" in output5_after_change, (
            f"Loop code change not detected after cached run! Got: {output5_after_change}"
        )

    def test_cfd_pattern_loop_change_with_intermediate_cells(self, nb_runner):
        """
        Reproduce the exact CFD demo pattern:
        - For-loop produces u, v, p
        - Post-processing cell computes vorticity and stream_function from u, v
        - Save cell uses stream_function, u, v, p, vorticity
        
        1. Run all cells
        2. Run save cell again (caches — our bug 2 fix)
        3. Modify the for-loop
        4. Run save cell → MUST detect change and re-execute loop + post-processing
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np\nfrom scipy.sparse.linalg import spsolve",
            # Cell 2: cash_on + debug
            "%cash_on\n%cash_debug on",
            # Cell 3: params
            "N = 10\ndx = 0.1\ndy = 0.1",
            # Cell 4: solver function
            (
                "import cash\n"
                "@cash.pure\n"
                "def apply_bc(u, v):\n"
                "    u[0] = 0; u[-1] = 1.0\n"
                "    v[0] = 0; v[-1] = 0\n"
                "    return u, v\n"
                "print('Solver functions defined')"
            ),
            # Cell 5: for-loop (the simulation)
            (
                "u = np.zeros(N)\n"
                "v = np.zeros(N)\n"
                "p = np.zeros(N)\n"
                "residual_history = []\n"
                "for step in range(500):\n"
                "    u = u + 0.001\n"
                "    v = v + 0.0005\n"
                "    p = u + v\n"
                "    u, v = apply_bc(u, v)\n"
                "    residual_history.append(np.max(np.abs(p)))\n"
                "print(f'Done: u_max={u.max():.4f}')"
            ),
            # Cell 6: post-processing (vorticity + stream_function)
            (
                "vorticity = np.zeros(N)\n"
                "vorticity[1:-1] = (v[2:] - v[:-2]) / (2 * dx)\n"
                "stream_function = np.cumsum(vorticity) * dx\n"
                "print(f'stream_max={stream_function.max():.6f}')"
            ),
            # Cell 7: save cell (like the user's cell 20)
            (
                "print(f'u_max={u.max():.4f}')\n"
                "print(f'stream_max={stream_function.max():.6f}')\n"
                "print(f'vorticity_range=[{vorticity.min():.6f}, {vorticity.max():.6f}]')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output7 = nb_runner.get_output(7)
        assert "u_max=" in output7
        assert "stream_max=" in output7

        # Run save cell a second time — should be cached (bug 2 fix)
        nb_runner.run_cell(7)
        output7_cached = nb_runner.get_output(7)
        assert "u_max=" in output7_cached

        # Now modify the for-loop: add u += 1 each iteration
        nb_runner.set_cell_source(5, (
            "u = np.zeros(N)\n"
            "v = np.zeros(N)\n"
            "p = np.zeros(N)\n"
            "residual_history = []\n"
            "for step in range(500):\n"
            "    u = u + 0.001\n"
            "    v = v + 0.0005\n"
            "    p = u + v\n"
            "    u, v = apply_bc(u, v)\n"
            "    u = u + 1\n"  # <-- NEW LINE
            "    residual_history.append(np.max(np.abs(p)))\n"
            "print(f'Done: u_max={u.max():.4f}')"
        ))

        # Run save cell — MUST detect the loop code change
        nb_runner.run_cell(7)
        output7_mod = nb_runner.get_output(7)
        assert "u_max=" in output7_mod

        # Extract u_max values and compare
        import re
        old_match = re.search(r'u_max=([\d.]+)', output7.strip())
        new_match = re.search(r'u_max=([\d.]+)', output7_mod.strip())
        assert old_match and new_match

        old_umax = float(old_match.group(1))
        new_umax = float(new_match.group(1))
        # With u += 1 added each iteration, u_max should be MUCH larger
        assert new_umax > old_umax * 10, (
            f"Loop code change not detected! old u_max={old_umax}, new u_max={new_umax}. "
            f"Expected much larger value after adding u += 1."
        )

    def test_loop_change_not_rerun_directly(self, nb_runner):
        """
        The case where the user modifies the loop but does NOT re-run it
        directly — just runs the downstream cell. The upstream checker must
        detect the code change in the .ipynb file and re-execute everything.
        """
        nb_runner.create_notebook([
            "import numpy as np",
            "%cash_on\n%cash_debug on",
            "N = 5",
            # Cell 4: for loop
            (
                "result = np.zeros(N)\n"
                "for i in range(100):\n"
                "    result = result + 1\n"
                "total = result.sum()\n"
                "print(f'total={total:.1f}')"
            ),
            # Cell 5: depends on result
            "value = result.max()\nprint(f'value={value:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output5 = nb_runner.get_output(5)
        assert "value=100.0" in output5

        # Modify loop WITHOUT running it — only change on disk
        nb_runner.set_cell_source(4, (
            "result = np.zeros(N)\n"
            "for i in range(200):\n"  # Changed from 100 to 200 iterations
            "    result = result + 1\n"
            "total = result.sum()\n"
            "print(f'total={total:.1f}')"
        ))

        # Run cell 5 — upstream checker should detect cell 4 changed on disk
        nb_runner.run_cell(5)
        output5_mod = nb_runner.get_output(5)
        assert "value=200.0" in output5_mod, (
            f"Loop code change (iterations) not detected! Got: {output5_mod}"
        )


class TestUnsavedEditDetection:
    """Test that loop changes are detected even when the .ipynb file hasn't been saved."""

    def _run_cell_with_unsaved_code(self, nb_runner, cell_num, new_code):
        """
        Execute a cell with new code in the kernel but write the OLD code
        back to disk, simulating an unsaved edit in VS Code.
        """
        idx = cell_num - 1
        old_code = nb_runner.nb.cells[idx].source

        # Temporarily set the new code and execute it
        nb_runner.nb.cells[idx].source = new_code
        nb_runner.nb.cells[idx].outputs = []
        nb_runner.client.execute_cell(nb_runner.nb.cells[idx], idx)

        # Write the OLD code back to disk (simulating unsaved state)
        nb_runner.nb.cells[idx].source = old_code
        nb_runner._save_notebook()

        # But keep the new code in memory (what VS Code shows)
        nb_runner.nb.cells[idx].source = new_code

    def test_unsaved_loop_edit_detected(self, nb_runner):
        """
        Reproduce the exact user bug:
        1. Run all cells with original loop code
        2. Edit the for-loop in VS Code (add u += 1) and execute it
        3. The .ipynb file on disk still has the old code
        4. Run a downstream cell -> should detect the unsaved edit and
           re-execute the intermediate cells with the new values
        """
        original_loop = (
            "u = np.zeros(N)\n"
            "v = np.zeros(N)\n"
            "p = np.zeros(N)\n"
            "for step in range(100):\n"
            "    u = u + 0.01\n"
            "    v = v + 0.005\n"
            "    p = u + v\n"
            "print(f'u_max={u.max():.4f}')"
        )
        modified_loop = (
            "u = np.zeros(N)\n"
            "v = np.zeros(N)\n"
            "p = np.zeros(N)\n"
            "for step in range(100):\n"
            "    u = u + 0.01\n"
            "    v = v + 0.005\n"
            "    p = u + v\n"
            "    u += 1\n"
            "print(f'u_max={u.max():.4f}')"
        )

        nb_runner.create_notebook([
            "import numpy as np",
            "%cash_on\n%cash_debug on",
            "N = 5\ndx = 0.1",
            original_loop,
            (
                "vorticity = (v[2:] - v[:-2]) / (2 * dx)\n"
                "stream = np.cumsum(u)\n"
                "print(f'stream_sum={stream.sum():.4f}')"
            ),
            "result = stream.sum()\nprint(f'result={result:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output6 = nb_runner.get_output(6)
        assert "result=" in output6

        # Simulate the unsaved edit: execute modified loop
        # but leave old code on disk
        self._run_cell_with_unsaved_code(nb_runner, 4, modified_loop)

        # Run the downstream cell -> should detect the unsaved edit
        nb_runner.run_cell(6)
        output6_mod = nb_runner.get_output(6)

        import re
        old_match = re.search(r'result=([\d.]+)', output6.strip())
        new_match = re.search(r'result=([\d.]+)', output6_mod.strip())
        assert old_match and new_match, f"Could not parse: old={output6}, new={output6_mod}"

        old_val = float(old_match.group(1))
        new_val = float(new_match.group(1))
        assert new_val != old_val, (
            f"Unsaved loop edit not detected! old={old_val}, new={new_val}. "
            f"The upstream checker should have detected the discrepancy."
        )

    def test_unsaved_loop_edit_cfd_pattern(self, nb_runner):
        """
        CFD-like pattern: loop -> post-processing -> save cell.
        User edits the loop and runs it, but doesn't save.
        Running the save cell should detect the stale intermediate values.
        """
        original_loop = (
            "data = np.zeros(N)\n"
            "for i in range(100):\n"
            "    data = data + 1\n"
            "total = data.sum()\n"
            "print(f'total={total:.1f}')"
        )
        modified_loop = (
            "data = np.zeros(N)\n"
            "for i in range(100):\n"
            "    data = data + 2\n"
            "total = data.sum()\n"
            "print(f'total={total:.1f}')"
        )

        nb_runner.create_notebook([
            "import numpy as np",
            "%cash_on\n%cash_debug on",
            "N = 5",
            original_loop,
            "processed = data * 2\nprint(f'processed={processed.sum():.1f}')",
            "final = processed.sum()\nprint(f'final={final:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output6 = nb_runner.get_output(6)
        assert "final=1000.0" in output6

        # Unsaved edit: execute modified loop, but disk has old code
        self._run_cell_with_unsaved_code(nb_runner, 4, modified_loop)

        # Run cell 6 -> should detect unsaved change and recompute
        nb_runner.run_cell(6)
        output6_mod = nb_runner.get_output(6)
        assert "final=2000.0" in output6_mod, (
            f"Unsaved loop edit not propagated! Got: {output6_mod}. "
            f"Expected final=2000.0 after loop increment changed from 1 to 2."
        )
