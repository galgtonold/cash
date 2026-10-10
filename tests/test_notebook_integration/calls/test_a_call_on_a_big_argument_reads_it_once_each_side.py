"""``y = normalize(x)`` reads ``x`` in full once before the call and once after.

The statement's in-place-change fingerprint and the cached call's own check
of its arguments each read ``x`` before and after: four reads on a first run,
19.1 s where plain Python took 0.89 s over a 257 MB frame. When the statement
is nothing but the call they share one read each side. A callee that changes
its argument is still seen doing it: it runs again on every run, and the
argument ends as it would without cash.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
DEFS = (
    "import time\nimport numpy as np\n"
    "x = np.arange(2_000_000.0)\n"
    "def normalize(a):\n    time.sleep(0.25)\n    return (a - a.mean()) / a.std()\n"
    "def bump(a):\n    time.sleep(0.25)\n    a[0] += 1\n    return 0\n"
)
# Installed afresh on every use, over the real function, with a fresh count:
# cash's modules outlive a test in a reused kernel.
COUNT = (
    "import cash.content_hashers as ch\n"
    "ch.hash_numpy = getattr(ch, 'real_hash_numpy', ch.hash_numpy)\n"
    "ch.real_hash_numpy = ch.hash_numpy\n"
    "ch.reads_of_x = []\n"
    "def counting(value, *a, ch=ch, **k):\n"
    "    if value is get_ipython().user_ns.get('x'):\n"
    "        ch.reads_of_x.append(1)\n"
    "    return ch.real_hash_numpy(value, *a, **k)\n"
    "ch.hash_numpy = counting\n"
)
READS = "len(__import__('cash.content_hashers').content_hashers.reads_of_x)"


def test_a_first_run_reads_the_argument_once_before_and_once_after(nb_runner):
    nb_runner.create_notebook([SETUP, DEFS, "y = normalize(x)", "print('Y', round(float(y[-1]), 4))"])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cell(3)
    assert nb_runner.peek(READS) == "2"
    nb_runner.run_cell(4)
    assert "Y 1.732" in nb_runner.get_output(4)


def test_a_call_changing_its_argument_runs_on_every_run(nb_runner):
    """Re-running the cell brings ``x`` back from its own cell first (the
    upstream check), so every run must see ``bump`` make its change again:
    served from the cache, ``x[0]`` would read 0.0."""
    nb_runner.create_notebook([SETUP, DEFS, "r = bump(x)", "print('X0', float(x[0]))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X0 1.0" in nb_runner.get_output(4)
    for _ in range(2):
        nb_runner.run_cells([3, 4])
        assert "X0 1.0" in nb_runner.get_output(4)
        assert "In-place mutation on: x" in nb_runner.get_output(3)
