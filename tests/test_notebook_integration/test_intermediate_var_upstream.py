"""
Test: Intermediate variables in upstream cells are properly scheduled for execution.

Reproduces the bug where:
1. Cell A computes: intermediate = f(x); result = g(intermediate)
2. Cell B uses `result` as input
3. After kernel restart, cell B is executed
4. Upstream checker schedules `result = g(intermediate)` for execution
5. But FAILS to also schedule `intermediate = f(x)` → NameError

Root cause: The backward scan in _simulate_and_find_changes should add
intermediate variables as needed when their consuming statement is scheduled.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(60)]


class TestIntermediateVariableUpstream:
    """Verify intermediate variables are properly auto-executed."""

    def test_intermediate_var_restored_after_restart(self, nb_runner):
        """
        When cell B needs `result` from cell A, and `result` depends on
        an intermediate `psi_flat` computed in the same cell, the upstream
        checker must also schedule the intermediate's producing statement.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np",
            # Cell 2: base data
            "x = np.array([1.0, 2.0, 3.0, 4.0])",
            # Cell 3: intermediate + derived
            (
                "intermediate = np.cumsum(x)\n"
                "result = intermediate.reshape(2, 2)"
            ),
            # Cell 4: uses result
            "print(f'Shape: {result.shape}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "Shape: (2, 2)" in output

        # Simulate kernel restart: reset cash state and clear namespace
        nb_runner.reset_cash_state()
        # Clear variables except imports
        nb_runner._run_async(
            nb_runner.client.kc._async_execute_interactive(
                "del x, intermediate, result", store_history=False
            )
        )

        # Re-run cell 4 only — upstream should auto-execute cells 2+3
        nb_runner.run_cell(4)
        output = nb_runner.get_output(4)
        assert "Shape: (2, 2)" in output

    def test_intermediate_var_chain_after_restart(self, nb_runner):
        """
        Longer chain: a → b → c → d, where cell uses d.
        After restart, all intermediates should be scheduled.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np",
            # Cell 2: chain of computations  
            (
                "a = np.ones(4)\n"
                "b = np.cumsum(a)\n"
                "c = b * 2\n"
                "d = c.sum()"
            ),
            # Cell 3: uses final result
            "print(f'Total: {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "Total: 20.0" in output

        # Simulate kernel restart
        nb_runner.reset_cash_state()
        nb_runner._run_async(
            nb_runner.client.kc._async_execute_interactive(
                "del a, b, c, d", store_history=False
            )
        )

        # Re-run cell 3 only
        nb_runner.run_cell(3)
        output = nb_runner.get_output(3)
        assert "Total: 20.0" in output

    def test_spsolve_pattern_after_restart(self, nb_runner):
        """
        Mimics the CFD demo pattern:
        - Cell computes psi_flat = spsolve(A, rhs)
        - Same cell derives stream_function = psi_flat.reshape(...)
        - Downstream cell uses stream_function
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            "import numpy as np\nfrom scipy.sparse import diags\nfrom scipy.sparse.linalg import spsolve",
            # Cell 2: setup
            "N = 5\nA = diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format='csc')\nrhs = np.ones(N)",
            # Cell 3: solve + reshape (intermediate pattern)
            (
                "psi_flat = spsolve(A, rhs)\n"
                "stream_function = psi_flat.reshape(1, N)"
            ),
            # Cell 4: use stream_function
            "print(f'SF shape: {stream_function.shape}, sum: {stream_function.sum():.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "SF shape:" in output

        # Simulate kernel restart
        nb_runner.reset_cash_state()
        nb_runner._run_async(
            nb_runner.client.kc._async_execute_interactive(
                "del N, A, rhs, psi_flat, stream_function", store_history=False
            )
        )

        # Re-run cell 4 only — should auto-execute cells 2+3 including psi_flat
        nb_runner.run_cell(4)
        output = nb_runner.get_output(4)
        assert "SF shape:" in output

    def test_cfd_pattern_multistatement_cell_with_prints(self, nb_runner):
        """
        Full CFD demo pattern with @cash.pure functions, factorized import,
        for loops, and multi-statement cells with intermediates.
        After restart, downstream cell should get stream_function via upstream.
        """
        nb_runner.create_notebook([
            # Cell 1: imports (mimics CFD cell 1)
            (
                "import cash\n"
                "import numpy as np\n"
                "from scipy import sparse\n"
                "from scipy.sparse.linalg import spsolve\n"
                "import time\n"
                "import os"
            ),
            # Cell 2: parameters (mimics CFD cell 4)
            (
                "Re = 100\n"
                "N = 5\n"
                "dx = 1.0 / (N - 1)\n"
                "dy = 1.0 / (N - 1)\n"
                "x = np.linspace(0, 1, N)\n"
                "y = np.linspace(0, 1, N)"
            ),
            # Cell 3: solver functions (mimics CFD cell 6 with @cash.pure)
            (
                "@cash.pure\n"
                "def build_laplacian_2d(n, dx, dy):\n"
                "    n2 = n * n\n"
                "    diags_arr = np.zeros((3, n2))\n"
                "    diags_arr[0, :] = 1.0 / dx**2\n"
                "    diags_arr[1, :] = -2.0 / dx**2 - 2.0 / dy**2\n"
                "    diags_arr[2, :] = 1.0 / dx**2\n"
                "    A = sparse.diags(diags_arr, [1, 0, -1], shape=(n2, n2), format='csc')\n"
                "    return A\n"
                "\n"
                "@cash.pure\n"
                "def apply_boundary_conditions(u, v, U_lid):\n"
                "    u[0, :] = 0; u[-1, :] = 0; u[:, 0] = 0; u[:, -1] = 0\n"
                "    v[0, :] = 0; v[-1, :] = 0; v[:, 0] = 0; v[:, -1] = 0\n"
                "    u[-1, :] = U_lid\n"
                "    return u, v\n"
                "\n"
                "print('Solver functions defined')"
            ),
            # Cell 4: build operators with factorized import (mimics CFD cell 8)
            (
                "t0 = time.time()\n"
                "A_laplacian = build_laplacian_2d(N, dx, dy)\n"
                "print(f'Built in {time.time() - t0:.3f}s')\n"
                "t0 = time.time()\n"
                "from scipy.sparse.linalg import factorized\n"
                "pressure_solve = factorized(A_laplacian)\n"
                "print(f'LU done in {time.time() - t0:.3f}s')"
            ),
            # Cell 5: simulation loop (mimics CFD cell 10)
            (
                "u = np.zeros((N, N))\n"
                "v = np.zeros((N, N))\n"
                "p = np.zeros((N, N))\n"
                "u, v = apply_boundary_conditions(u, v, 1.0)\n"
                "residual_history = []\n"
                "for step in range(10):\n"
                "    u = u * 0.99\n"
                "    u, v = apply_boundary_conditions(u, v, 1.0)\n"
                "    residual_history.append(np.max(np.abs(u)))\n"
                "print(f'Sim done: {len(residual_history)} steps')"
            ),
            # Cell 6: post-processing with intermediates (mimics CFD cell 12)
            (
                "t0 = time.time()\n"
                "vorticity = np.zeros((N, N))\n"
                "vorticity[1:-1, 1:-1] = (\n"
                "    (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) -\n"
                "    (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)\n"
                ")\n"
                "print(f'Vorticity computed in {time.time()-t0:.3f}s')\n"
                "t0 = time.time()\n"
                "psi_flat = spsolve(A_laplacian, -vorticity.flatten())\n"
                "stream_function = psi_flat.reshape((N, N))\n"
                "print(f'Stream function computed in {time.time()-t0:.3f}s')\n"
                "t0 = time.time()\n"
                "kinetic_energy = 0.5 * (u**2 + v**2)\n"
                "total_KE = np.sum(kinetic_energy) * dx * dy\n"
                "print(f'KE: {total_KE:.6f}')"
            ),
            # Cell 7: downstream cell (mimics the save cell)
            (
                "print(f'stream_function shape: {stream_function.shape}')\n"
                "print(f'u shape: {u.shape}')\n"
                "print(f'vorticity range: [{vorticity.min():.4f}, {vorticity.max():.4f}]')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(7)
        assert "stream_function shape:" in output
        assert "u shape:" in output

        # Simulate kernel restart: clear everything
        nb_runner.reset_cash_state()
        nb_runner._run_async(
            nb_runner.client.kc._async_execute_interactive(
                "for _v in ['Re','N','dx','dy','x','y','A_laplacian','pressure_solve',"
                "'u','v','p','residual_history','t0',"
                "'vorticity','psi_flat','stream_function','kinetic_energy','total_KE',"
                "'step','build_laplacian_2d','apply_boundary_conditions','factorized']:\n"
                "    try:\n"
                "        exec(f'del {_v}')\n"
                "    except NameError:\n"
                "        pass",
                store_history=False
            )
        )

        # Re-run cell 7 only — should trigger full upstream chain
        nb_runner.run_cell(7)
        output = nb_runner.get_output(7)
        assert "stream_function shape:" in output, f"Expected stream_function in output, got: {output}"
        assert "u shape:" in output

    def test_cfd_pattern_actual_kernel_restart(self, nb_runner):
        """
        Same as above but with ACTUAL kernel restart (not just state clearing).
        This matches the user's exact scenario: kernel restart -> run imports + cash_on -> run downstream cell.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            (
                "import cash\n"
                "import numpy as np\n"
                "from scipy import sparse\n"
                "from scipy.sparse.linalg import spsolve\n"
                "import time\n"
                "import os"
            ),
            # Cell 2: cash_on
            "%cash_on",
            # Cell 3: parameters
            (
                "Re = 100\n"
                "N = 5\n"
                "dx = 1.0 / (N - 1)\n"
                "dy = 1.0 / (N - 1)\n"
                "x = np.linspace(0, 1, N)\n"
                "y = np.linspace(0, 1, N)"
            ),
            # Cell 4: solver functions with @cash.pure
            (
                "@cash.pure\n"
                "def build_laplacian_2d(n, dx, dy):\n"
                "    n2 = n * n\n"
                "    diags_arr = np.zeros((3, n2))\n"
                "    diags_arr[0, :] = 1.0 / dx**2\n"
                "    diags_arr[1, :] = -2.0 / dx**2 - 2.0 / dy**2\n"
                "    diags_arr[2, :] = 1.0 / dx**2\n"
                "    A = sparse.diags(diags_arr, [1, 0, -1], shape=(n2, n2), format='csc')\n"
                "    return A\n"
                "\n"
                "@cash.pure\n"
                "def apply_boundary_conditions(u, v, U_lid):\n"
                "    u[0, :] = 0; u[-1, :] = 0; u[:, 0] = 0; u[:, -1] = 0\n"
                "    v[0, :] = 0; v[-1, :] = 0; v[:, 0] = 0; v[:, -1] = 0\n"
                "    u[-1, :] = U_lid\n"
                "    return u, v\n"
                "\n"
                "print('Solver functions defined')"
            ),
            # Cell 5: build operators with factorized import
            (
                "t0 = time.time()\n"
                "A_laplacian = build_laplacian_2d(N, dx, dy)\n"
                "print(f'Built in {time.time() - t0:.3f}s')\n"
                "t0 = time.time()\n"
                "from scipy.sparse.linalg import factorized\n"
                "pressure_solve = factorized(A_laplacian)\n"
                "print(f'LU done in {time.time() - t0:.3f}s')"
            ),
            # Cell 6: simulation loop
            (
                "u = np.zeros((N, N))\n"
                "v = np.zeros((N, N))\n"
                "p = np.zeros((N, N))\n"
                "u, v = apply_boundary_conditions(u, v, 1.0)\n"
                "residual_history = []\n"
                "for step in range(10):\n"
                "    u = u * 0.99\n"
                "    u, v = apply_boundary_conditions(u, v, 1.0)\n"
                "    residual_history.append(np.max(np.abs(u)))\n"
                "print(f'Sim done: {len(residual_history)} steps')"
            ),
            # Cell 7: post-processing with intermediates
            (
                "t0 = time.time()\n"
                "vorticity = np.zeros((N, N))\n"
                "vorticity[1:-1, 1:-1] = (\n"
                "    (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) -\n"
                "    (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)\n"
                ")\n"
                "print(f'Vorticity computed in {time.time()-t0:.3f}s')\n"
                "t0 = time.time()\n"
                "psi_flat = spsolve(A_laplacian, -vorticity.flatten())\n"
                "stream_function = psi_flat.reshape((N, N))\n"
                "print(f'Stream function computed in {time.time()-t0:.3f}s')\n"
                "t0 = time.time()\n"
                "kinetic_energy = 0.5 * (u**2 + v**2)\n"
                "total_KE = np.sum(kinetic_energy) * dx * dy\n"
                "print(f'KE: {total_KE:.6f}')"
            ),
            # Cell 8: downstream cell that uses stream_function
            (
                "print(f'stream_function shape: {stream_function.shape}')\n"
                "print(f'u shape: {u.shape}')\n"
                "print(f'vorticity range: [{vorticity.min():.4f}, {vorticity.max():.4f}]')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(8)
        assert "stream_function shape:" in output
        assert "u shape:" in output

        # ACTUAL kernel restart
        import asyncio
        nb_runner.restart()

        # Re-inject notebook path (lost after restart)
        nb_runner._inject_notebook_path()

        # Re-run cells 1 (imports) and 2 (cash_on) to set up the environment
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)

        # Now run cell 8 directly — upstream should auto-execute everything needed
        nb_runner.run_cell(8)
        output = nb_runner.get_output(8)
        assert "stream_function shape:" in output, f"Expected stream_function in output, got: {output}"
        assert "u shape:" in output
