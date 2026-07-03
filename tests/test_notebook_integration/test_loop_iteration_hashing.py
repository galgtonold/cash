"""CAS-86: per-iteration loop caching must discriminate iterations by FULL
content — a sampled hash keyed two iterations over arrays that agreed in the
sample onto one entry, producing a wrong result on the very first run."""

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.loops]


def test_ndarray_iterations_outside_sample_do_not_collide(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np\n"
        "batches = [np.zeros(2000), np.zeros(2000)]\n"
        "batches[1][1000] = 5.0",
        "for b in batches:\n"
        "    s = float(b.sum())\n"
        "    print('s=' + str(s))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    # Two iterations differing only in the middle of the array: the second
    # must print 5.0, not a replay of iteration 1's cached 0.0.
    assert "s=0.0" in out and "s=5.0" in out, (
        f"loop-iteration hash collision on first run; cell2 printed {out!r}"
    )


def test_identical_loop_second_run_still_cached(nb_runner):
    """Control: full-content hashing must not break iteration cache HITS."""
    nb_runner.create_notebook([
        "import numpy as np\nbatches = [np.ones(2000), np.full(2000, 2.0)]",
        "tot = 0.0\nfor b in batches:\n"
        "    s = float(b.sum())\n"
        "    tot += s\nprint('tot=' + str(tot))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "tot=6000.0" in nb_runner.get_output(2)

    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "tot=6000.0" in out, f"second run wrong: {out!r}"
