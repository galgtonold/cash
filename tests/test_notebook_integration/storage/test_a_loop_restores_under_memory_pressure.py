"""A loop's iterations still restore when the machine's memory is full.

A ten-iteration loop filling a dict -- the quickstart's own
headline shape, ``prices[ticker] = fetch_and_model(ticker)`` -- restored
exactly ONE of its ten iterations on every identical re-run:

    LOOP x10: loaded[r] = load_run(r)  - 1 cached (saved 0.16s), 9 ran (6.26s)

and the user found it came back to ten only "when the cell ABOVE is re-run
immediately before" -- a condition they called impossible to guess.

It is the RAM tier's memory-pressure check. It fires on every tenth write and
reads the WHOLE MACHINE's memory; with five people and five oracle kernels on
one box that sat above 90%, and the eviction loop dropped entries until the
machine fell under 81%, which cash cannot make happen, so it emptied the tier.
Landing on the loop's ninth write, it dropped iterations one to nine and kept
the tenth. Re-running the cell above added writes and moved the check off the
loop. And the cascade reported separately follows from it: with nine of
ten iterations re-running, the dict took a new lineage every run, every key
downstream changed with it, and the perpetual-miss guard correctly retired ten
statements as "unstable key".

On a quiet machine this loop restores ten of ten on the old code as well; the
pressure below is what makes it one of ten. The machine's memory is reported
at 95% by patching the backend's psutil, so what cash decides does not depend
on what else is running.

The tier still gives back its SHARE, and holds flat after that: the least
valuable bytes go, and those are the loop's statement entries (each holds the
growing dict) long before the tiny results of the ``work`` calls inside them.
So an identical re-run may run a few iterations again, each served its
``work(k)`` from the call cache in milliseconds -- "6 cached, 4 ran (0.02s),
sub-call work(k): 4/4 hit" is the tier doing its job, not the bug. What the
bug did was re-run the WORK, so that is what is counted, with a counter the
cached function cannot replay.

Nothing here goes past RAM (``RAM_ONLY``): an iteration that also reached disk
restores from there whatever the RAM tier did, and whether one does depends on
it costing more than the disk tier's 0.1 s floor -- on how busy the machine
is. Under load every iteration did, and the test passed on the old,
tier-emptying code too; on a quiet machine some did not. Kept in RAM, the old
code re-runs all ten ``work`` calls on every re-run.

The unit twin is ``tests/test_backends/test_ram_tier_sheds_its_share.py``.
"""

import pytest

# A fresh kernel: SETUP patches cash's memory backend in place, and a warm
# kernel's reset between tests keeps cash's modules but clears the names the
# patch's lambda reads. On a warm kernel every later test on the worker then
# got a NameError out of the RAM tier's pressure check on every tenth write
# (measured: the next test's `m.psutil.virtual_memory()` raised); a fresh
# kernel takes the patch with it, and starts this test from nothing earlier
# tests left behind.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(600), pytest.mark.fresh_kernel]

#: No entry is worth a disk write: a restore would have to save all of the
#: compute (``PersistencePolicy.pays_to_restore``), so the RAM tier is the only
#: place anything is kept and the test sees what IT does.
RAM_ONLY = "cash.configure(min_cache_savings_pct=1.0)"

SETUP = (
    f"import cash\n%cash_on\n%cash_badge print\n{RAM_ONLY}\nimport os\nimport numpy as np\n"
    "import psutil as _ps, types as _ty\n"
    "import cash.backends.memory_backend as _mb\n"
    "_tot = _ps.virtual_memory().total\n"
    "_mb.psutil = _ty.SimpleNamespace(virtual_memory="
    "lambda: _ty.SimpleNamespace(percent=95.0, total=_tot))"
)

KEYS = 'KEYS = ["k%02d" % i for i in range(10)]'


def _loop(counter) -> str:
    """The loop, with ``work`` appending a byte to *counter* each time it really runs."""
    return LOOP.replace("def work(k):\n", f"def work(k):\n    _count({str(counter)!r})\n", 1)


#: ``os.write``, not ``open``: an ``open`` write is an effect a hit replays.
COUNT = """def _count(path):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, b"x")
    os.close(fd)
"""

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


def test_a_ten_iteration_loop_redoes_none_of_its_work_under_pressure(nb_runner, tmp_path):
    counter = tmp_path / "work_calls.log"
    nb_runner.create_notebook([SETUP, KEYS, COUNT + "\n" + _loop(counter)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(3)
    assert counter.stat().st_size == 10, "the first run should run work() for every key"

    # The first re-run under pressure gives back the tier's share; what matters
    # is where it settles, which is what the user lived with.
    nb_runner.run_cell(3)
    before = counter.stat().st_size
    nb_runner.run_cell(3)
    ran = counter.stat().st_size - before
    raw = nb_runner.get_raw_output(3)

    assert _sum(nb_runner.get_output(3)) == _sum(first), raw
    assert ran == 0, (
        f"an identical re-run under memory pressure ran work() {ran} times instead of restoring it:\n" + raw
    )
    loop = next(line for line in raw.splitlines() if "LOOP x10" in line)
    assert "cached" in loop, "the RAM tier emptied itself: no iteration of the loop restored:\n" + raw
