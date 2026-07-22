"""seed(None) must warn that cached values below it cannot be fresh.

An entropy reseed asks for a different stream every run, but cash caches
downstream results. A value whose lineage is source-derived rather than
stream-derived -- a model from an in-place ``fit()`` and everything that reads
it -- is frozen from the run that first computed it. So the user gets a new
stream and a stale cached value describing the previous one, and they disagree
silently. No cache key resolves it (making the value fresh-per-run stops it
converging), so the honest fix is to warn.

The warning is advisory: behaviour is unchanged, and a fixed-int seed or a
``# @cash:no-cache`` annotation is the way out, both named in the message.
"""
import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"


def _cell_warnings(nb_runner, cell_num):
    """stderr text of a cell (where CashWarning lands under the kernel)."""
    return nb_runner.get_output(cell_num, filter_debug=False)


@pytest.mark.timeout(180)
def test_entropy_reseed_warns_about_frozen_downstream(nb_runner):
    nb_runner.create_notebook([
        C_ON,
        "import numpy as np\nnp.random.seed(None)\nx = np.random.rand(3)\nprint('X', float(x.sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = _cell_warnings(nb_runner, 2)
    assert "seed(None)" in out, f"no entropy-reseed warning emitted:\n{out}"
    assert "no-cache" in out or "fixed" in out, (
        f"the warning must name a way out:\n{out}"
    )


@pytest.mark.timeout(180)
def test_fixed_seed_does_not_warn(nb_runner):
    """A reproducible seed is not an entropy reseed and must stay silent."""
    nb_runner.create_notebook([
        C_ON,
        "import numpy as np\nnp.random.seed(42)\nx = np.random.rand(3)\nprint('X', float(x.sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = _cell_warnings(nb_runner, 2)
    assert "seed(None)" not in out, (
        f"a fixed-int seed must not trigger the entropy-reseed warning:\n{out}"
    )
