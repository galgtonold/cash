"""``y = normalize(x)`` reads ``x`` in full once before the call and once after.

The statement's in-place-change fingerprint and the cached call's own check
of its arguments each read ``x`` before and after: four reads on a first run,
19.1 s where plain Python took 0.89 s over a 257 MB frame. When the statement
is nothing but the call they share one read each side. A callee that changes
its NumPy argument is stored with the state it leaves the argument in, and a
hit writes that state back into the live array: the argument ends as it
would without cash.
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
# Put back what COUNT replaced: left in a reused kernel, `counting` is a
# function of the user's whose text cash cannot read, running inside every
# later statement that hashes an array.
UNCOUNT = (
    "import cash.content_hashers as ch\n"
    "if hasattr(ch, 'real_hash_numpy'):\n"
    "    ch.hash_numpy = ch.real_hash_numpy\n"
    "    del ch.real_hash_numpy\n"
)
READS = "len(__import__('cash.content_hashers').content_hashers.reads_of_x)"


def test_a_first_run_reads_the_argument_once_before_and_once_after(nb_runner):
    nb_runner.create_notebook([SETUP, DEFS, "y = normalize(x)", "print('Y', round(float(y[-1]), 4))"])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    try:
        nb_runner.run_cell(3)
        assert nb_runner.peek(READS) == "2"
        nb_runner.run_cell(4)
        assert "Y 1.732" in nb_runner.get_output(4)
    finally:
        nb_runner.peek(f"exec({UNCOUNT!r})")


def test_a_call_changing_its_argument_is_served_and_writes_the_change_back(nb_runner):
    """Philipp's decision of 2026-11-03: a statement whose call changes its
    input in place is stored, and a hit writes the state the run left the
    input in back into the live object, for NumPy arrays, pandas frames and
    plain lists, dicts and sets; anything else keeps running every time.

    ``x`` is a NumPy array, so ``r = bump(x)`` is served on every run after
    the first, and ``x[0]`` reads 1.0 as after a run from the top: not 0.0
    (served without the write) nor 2.0 (bumped twice)."""
    nb_runner.create_notebook([SETUP, DEFS, "r = bump(x)", "print('X0', float(x[0]))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X0 1.0" in nb_runner.get_output(4)
    for _ in range(2):
        nb_runner.run_cells([3, 4])
        assert "X0 1.0" in nb_runner.get_output(4)
        assert "CACHED: r = bump(x)" in nb_runner.get_output(3), nb_runner.get_output(3)
        assert nb_runner.peek("float(x[0])") == "1.0"
