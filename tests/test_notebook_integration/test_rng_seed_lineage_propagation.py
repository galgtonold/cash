"""A seed change must invalidate cached statements DOWNSTREAM of a draw (ADR-018).

The RNG epoch (CAS-223) was folded into a draw's cache key but never into its
output lineage, so a draw recomputed on a re-seed while its output variable's
lineage stayed constant — every cached consumer keyed on the unchanged lineage
and served a stale value, even on a full Run-All. Modelling the RNG as a hidden
lineage variable (a seed produces it, a draw reads it) makes the seed dependency
propagate through the ordinary input-lineage machinery.

Oracle = the current cell sources run top-to-bottom with no cash.
"""
import re

import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
DATA = "import numpy as np\nnp.random.seed(7)\nN = 4000\na = np.random.rand(N)"


def _num(runner, n, tag):
    m = re.search(tag + r"\s+([0-9.eE+-]+)", runner.get_output(n))
    return float(m.group(1)) if m else None


def _oracle_sum(seed, transform):
    import numpy as np
    np.random.seed(seed)
    a = np.random.rand(4000)
    return float(transform(a))


@pytest.mark.timeout(180)
def test_seed_edit_propagates_downstream_on_run_all(nb_runner):
    """Edit the seed; a cached downstream consumer must refresh on Run-All."""
    nb_runner.create_notebook([
        C_ON, DATA,
        "# @cash:persist\nroll = float(np.convolve(a, np.ones(20) / 20, 'valid').sum())\nprint('ROLL', round(roll, 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, DATA.replace("np.random.seed(7)", "np.random.seed(123)"))
    nb_runner.run_all()
    got = _num(nb_runner, 3, "ROLL")
    want = _oracle_sum(123, lambda a: __import__("numpy").convolve(a, __import__("numpy").ones(20) / 20, "valid").sum())
    assert got == pytest.approx(want, rel=1e-9), (
        f"downstream consumer stale after seed edit: cash={got} oracle={want}"
    )


@pytest.mark.timeout(180)
def test_seed_edit_propagates_on_isolated_downstream_rerun(nb_runner):
    """Edit the seed, re-run ONLY the downstream consumer (exercises the simulator)."""
    nb_runner.create_notebook([
        C_ON, DATA,
        "# @cash:persist\ns = float(a.sum())\nprint('S', round(s, 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, DATA.replace("np.random.seed(7)", "np.random.seed(123)"))
    nb_runner.run_cell(3)  # data cell NOT re-run directly
    got = _num(nb_runner, 3, "S")
    want = _oracle_sum(123, lambda a: a.sum())
    assert got == pytest.approx(want, rel=1e-9), (
        f"downstream consumer stale on isolated re-run after seed edit: cash={got} oracle={want}"
    )


@pytest.mark.timeout(180)
def test_seed_edit_propagates_two_hops(nb_runner):
    """The seed dependency must reach a transitive (two-hop) consumer."""
    nb_runner.create_notebook([
        C_ON, DATA,
        "# @cash:persist\nb = a * 2.0",
        "# @cash:persist\nc = float(b.sum())\nprint('C', round(c, 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, DATA.replace("np.random.seed(7)", "np.random.seed(123)"))
    nb_runner.run_all()
    got = _num(nb_runner, 4, "C")
    want = _oracle_sum(123, lambda a: (a * 2.0).sum())
    assert got == pytest.approx(want, rel=1e-9), (
        f"two-hop consumer stale after seed edit: cash={got} oracle={want}"
    )


@pytest.mark.timeout(180)
def test_no_seed_edit_still_hits(nb_runner):
    """Sanity: with no edit, the downstream consumer still restores (no over-invalidation)."""
    nb_runner.create_notebook([
        C_ON, DATA,
        "# @cash:persist\ns = float(a.sum())\nprint('S', round(s, 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _num(nb_runner, 3, "S")
    nb_runner.run_cell(3)
    assert _num(nb_runner, 3, "S") == pytest.approx(first, abs=1e-9) == pytest.approx(_oracle_sum(7, lambda a: a.sum()), rel=1e-9)
