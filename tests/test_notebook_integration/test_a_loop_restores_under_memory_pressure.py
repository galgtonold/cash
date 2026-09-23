"""A loop's iterations still restore when the machine's memory is full.

Round 27, r27s4: a ten-iteration loop filling a dict -- the quickstart's own
headline shape, ``prices[ticker] = fetch_and_model(ticker)`` -- restored
exactly ONE of its ten iterations on every identical re-run:

    LOOP x10: loaded[r] = load_run(r)  - 1 cached (saved 0.16s), 9 ran (6.26s)

and the tester found it came back to ten only "when the cell ABOVE is re-run
immediately before" -- a condition they called impossible to guess.

It is the RAM tier's memory-pressure check. It fires on every tenth write and
reads the WHOLE MACHINE's memory; with five testers and five oracle kernels on
one box that sat above 90%, and the eviction loop dropped entries until the
machine fell under 81%, which cash cannot make happen, so it emptied the tier.
Landing on the loop's ninth write, it dropped iterations one to nine and kept
the tenth. Re-running the cell above added writes and moved the check off the
loop. And the cascade the tester filed separately follows from it: with nine of
ten iterations re-running, the dict took a new lineage every run, every key
downstream changed with it, and the perpetual-miss guard correctly retired ten
statements as "unstable key".

On a quiet machine this loop restores ten of ten on the old code as well; the
pressure below is what makes it one of ten. The machine's memory is reported
at 95% by patching the backend's psutil, so what cash decides does not depend
on what else is running.

The unit twin is ``tests/test_backends/test_ram_tier_sheds_its_share.py``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

# Restoring from cache, the standing caveat: the canonical explanation is in
# `test_loop_edit_rerun_matrix.py`. Only a failure saying "re-ran" is retried.
LOAD_SENSITIVE = pytest.mark.flaky(reruns=2, reruns_delay=5, only_rerun=["re-ran"])

SETUP = (
    "import cash\n%cash_on\n%cash_badge print\nimport numpy as np\n"
    "import psutil as _ps, types as _ty\n"
    "import cash.backends.memory_backend as _mb\n"
    "_tot = _ps.virtual_memory().total\n"
    "_mb.psutil = _ty.SimpleNamespace(virtual_memory="
    "lambda: _ty.SimpleNamespace(percent=95.0, total=_tot))"
)

KEYS = 'KEYS = ["k%02d" % i for i in range(10)]'

# Each iteration a large matmul, so every one is unambiguously worth caching:
# at a few milliseconds the cost floor, not the pressure, decides, and the
# test would measure the wrong thing.
LOOP = """def work(k):
    n = 1800 + int(k[1:])
    a = np.random.default_rng(int(k[1:])).random((n, n))
    return float((a @ a).sum())

res = {}
for k in KEYS:
    res[k] = work(k)
print("SUM", round(sum(res.values()), 3))
"""


def _sum(out):
    """The printed SUM alone; the output also carries the badge."""
    import re

    m = re.search(r"SUM (\S+)", out or "")
    return m.group(1) if m else None


@LOAD_SENSITIVE
def test_a_ten_iteration_loop_restores_all_ten_under_pressure(nb_runner):
    nb_runner.create_notebook([SETUP, KEYS, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(3)

    # The first re-run under pressure gives back the tier's share; what matters
    # is where it settles, which is what the tester lived with.
    nb_runner.run_cell(3)
    nb_runner.run_cell(3)
    raw = nb_runner.get_raw_output(3)

    assert _sum(nb_runner.get_output(3)) == _sum(first), raw
    assert "10 cached" in raw, (
        "an identical re-run under memory pressure re-ran iterations of the loop instead of restoring all ten:\n" + raw
    )
