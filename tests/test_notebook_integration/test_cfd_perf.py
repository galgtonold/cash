"""
Performance test: Measure caching overhead on CFD-style compute loops.

Compares wall-clock time with caching enabled vs disabled for a loop
that does heavy numpy computation with periodic prints.  The loop contains
in-place mutations (``residual_history.append``) so it is treated as
uncacheable, meaning the only overhead is the statement analysis and the
exec wrapper — NOT the TeeWriter or cache I/O.

These tests catch performance regressions before they reach production.
"""
import pytest
import re
import time

pytestmark = [pytest.mark.timeout(120)]


# ---------------------------------------------------------------------------
# Notebook cells (mirror the user's real CFD simulation notebook)
# ---------------------------------------------------------------------------

SETUP_CELLS = [
    # Cell 1: imports
    """\
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized
import time
""",
    # Cell 2: parameters (smaller grid/steps for testing, but enough to measure overhead)
    """\
Re = 100
U_lid = 1.0
L = 1.0
nu = U_lid * L / Re
rho = 1.0
N = 41
dx = L / (N - 1)
dy = L / (N - 1)
dt = 0.001
n_steps = 5000
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
""",
    # Cell 3: solver functions
    """\
def build_laplacian_2d(n, dx, dy):
    n2 = n * n
    diags = np.zeros((5, n2))
    diags[2, :] = -2.0 / dx**2 - 2.0 / dy**2
    diags[1, :] = 1.0 / dx**2
    diags[3, :] = 1.0 / dx**2
    diags[0, :] = 1.0 / dy**2
    diags[4, :] = 1.0 / dy**2
    offsets = [n, 1, 0, -1, -n]
    A = sparse.diags(diags, offsets, shape=(n2, n2), format='csc')
    return A

def compute_divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    )
    return div

def apply_boundary_conditions(u, v, U_lid):
    u[0, :] = 0;  u[-1, :] = 0;  u[:, 0] = 0;  u[:, -1] = 0
    v[0, :] = 0;  v[-1, :] = 0;  v[:, 0] = 0;  v[:, -1] = 0
    u[-1, :] = U_lid
    return u, v
""",
    # Cell 4: build operators
    """\
A_laplacian = build_laplacian_2d(N, dx, dy)
pressure_solve = factorized(A_laplacian)
""",
]

# The CFD loop cell — this is the one we're measuring
CFD_LOOP_CELL = """\
u = np.zeros((N, N))
v = np.zeros((N, N))
p = np.zeros((N, N))
u, v = apply_boundary_conditions(u, v, U_lid)

t_start = time.time()
residual_history = []

for step in range(n_steps):
    u_adv = u.copy()
    v_adv = v.copy()
    
    u_adv[1:-1, 1:-1] = (u[1:-1, 1:-1]
        - dt * u[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        - dt * v[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        + dt * nu * (
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2 +
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        ))
    
    v_adv[1:-1, 1:-1] = (v[1:-1, 1:-1]
        - dt * u[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        - dt * v[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        + dt * nu * (
            (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2 +
            (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
        ))
    
    div = compute_divergence(u_adv, v_adv, dx, dy)
    rhs = (rho / dt) * div.flatten()
    p_flat = pressure_solve(rhs)
    p = p_flat.reshape((N, N))
    
    u[1:-1, 1:-1] = u_adv[1:-1, 1:-1] - (dt / rho) * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    v[1:-1, 1:-1] = v_adv[1:-1, 1:-1] - (dt / rho) * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    
    u, v = apply_boundary_conditions(u, v, U_lid)
    
    residual = np.sqrt(np.mean(div[1:-1, 1:-1]**2))
    residual_history.append(residual)
    
    if step % 1 == 0:
        print(f"  Step {step:4d}/{n_steps}: residual={residual:.2e}, "
              f"|u|_max={np.max(np.abs(u)):.4f}, |v|_max={np.max(np.abs(v)):.4f}")

wall_time = time.time() - t_start
print(f"Simulation complete in {wall_time:.2f}s")
print(f"Final residual: {residual_history[-1]:.2e}")
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_inner_time(output: str, fallback: float) -> float:
    """Extract 'Simulation complete in X.XXs' from cell output."""
    match = re.search(r'Simulation complete in (\d+\.\d+)s', output)
    return float(match.group(1)) if match else fallback


def _run_cfd(nb_runner, *, with_cash: bool) -> tuple[float, float, str]:
    """Run the full CFD notebook and return (wall_time, inner_time, output)."""
    nb_runner.create_notebook(SETUP_CELLS + [CFD_LOOP_CELL])
    nb_runner.start_kernel(with_cash=with_cash)

    for i in range(1, len(SETUP_CELLS) + 1):
        nb_runner.run_cell(i)

    loop_cell = len(SETUP_CELLS) + 1
    t0 = time.time()
    nb_runner.run_cell(loop_cell)
    wall = time.time() - t0

    output = nb_runner.get_output(loop_cell)
    assert "Simulation complete" in output, (
        f"Loop cell didn't complete (cash={'ON' if with_cash else 'OFF'}). "
        f"Output: {output[:500]}"
    )

    inner = _extract_inner_time(output, wall)
    nb_runner.shutdown()
    return wall, inner, output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cfd_loop_overhead(nb_runner):
    """Caching overhead for the CFD loop must stay within budget.

    Runs the 5 000-step Navier-Stokes loop with caching ON, then OFF,
    and asserts total wall-clock overhead is <20 % or <2 s absolute.
    Also checks inner (pure-Python) overhead as a sanity check.
    """
    wall_with, inner_with, _ = _run_cfd(nb_runner, with_cash=True)
    wall_without, inner_without, _ = _run_cfd(nb_runner, with_cash=False)

    overhead = wall_with - wall_without
    pct = (overhead / wall_without * 100) if wall_without > 0 else 0
    inner_overhead = inner_with - inner_without

    print(f"\n{'='*60}")
    print("CFD LOOP PERFORMANCE (5000 steps, N=41)")
    print(f"{'='*60}")
    print(f"  WITH caching:    {wall_with:.2f}s total, {inner_with:.2f}s inner")
    print(f"  WITHOUT caching: {wall_without:.2f}s total, {inner_without:.2f}s inner")
    print(f"  Total overhead:  {overhead:.2f}s ({pct:.1f}%)")
    print(f"  Inner overhead:  {inner_overhead:.2f}s")
    print(f"{'='*60}")

    # Total wall-clock overhead: <20% or <2 s (whichever is larger)
    assert overhead < max(wall_without * 0.20, 2.0), (
        f"Caching overhead too high! {overhead:.2f}s on {wall_without:.2f}s base "
        f"({pct:.1f}%). Must be <20% or <2s."
    )

    # Inner time (pure computation, no I/O) should show near-zero overhead.
    # The skip_capture fast-path means the exec runs with the real sys.stdout
    # -- identical to no-caching.  Allow 0.5 s tolerance for OS scheduling.
    assert inner_overhead < 0.5, (
        f"Inner computation overhead too high: {inner_overhead:.2f}s. "
        f"Expected near-zero because skip_capture bypasses TeeWriter."
    )


def test_cfd_loop_rerun_no_regression(nb_runner):
    """Running the loop a second time in the same kernel must not regress.

    Since the loop has mutations (residual_history.append), it's always
    COMPUTED, never RESTORED.  The second run should therefore take about
    the same time as the first (no extra overhead from stale lineage etc.).
    """
    nb_runner.create_notebook(SETUP_CELLS + [CFD_LOOP_CELL])
    nb_runner.start_kernel(with_cash=True)

    for i in range(1, len(SETUP_CELLS) + 1):
        nb_runner.run_cell(i)

    loop_cell = len(SETUP_CELLS) + 1

    # First run
    t0 = time.time()
    nb_runner.run_cell(loop_cell)
    first_run = time.time() - t0
    out1 = nb_runner.get_output(loop_cell)
    assert "Simulation complete" in out1

    # Second run (same cell)
    t0 = time.time()
    nb_runner.run_cell(loop_cell)
    second_run = time.time() - t0
    out2 = nb_runner.get_output(loop_cell)
    assert "Simulation complete" in out2

    inner1 = _extract_inner_time(out1, first_run)
    inner2 = _extract_inner_time(out2, second_run)

    print(f"\n{'='*60}")
    print("CFD LOOP RE-RUN (5000 steps, N=41)")
    print(f"{'='*60}")
    print(f"  First run:  {first_run:.2f}s total, {inner1:.2f}s inner")
    print(f"  Second run: {second_run:.2f}s total, {inner2:.2f}s inner")
    print(f"  Delta:      {second_run - first_run:.2f}s")
    print(f"{'='*60}")

    # Second run should not be more than 50% slower than first
    assert second_run < first_run * 1.5 + 1.0, (
        f"Second run regressed: {second_run:.2f}s vs first {first_run:.2f}s"
    )

    nb_runner.shutdown()
