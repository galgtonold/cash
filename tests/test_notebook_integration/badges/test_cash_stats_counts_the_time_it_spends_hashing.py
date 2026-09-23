"""Time cash spends keying a call is reported as overhead, not as your compute.

A paired Restart & Run All measured cash 370 s slower than
plain Jupyter while ``%cash_stats`` reported 210 s of overhead. Their loop
called a helper with a ~1 GB argument, and keying it hashes that argument --
inside the statement, so it counted as the user's own compute and cancelled
out of the overhead.

Here the hashing is made slow on purpose (pickling the argument sleeps), so
the tax is a known number rather than a machine-dependent one.
"""

import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\nimport time",
    "class Slow:\n"
    "    def __init__(self, n):\n        self.n = n\n"
    "    def __reduce__(self):\n        time.sleep(0.6)\n        return (Slow, (self.n,))\n"
    "def work(p):\n    return p.n * 2\n"
    "out = work(Slow(21))\nprint('OUT', out)",
    "%cash_stats json",
]


def test_the_hashing_tax_is_reported_as_overhead(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT 42" in nb_runner.get_output(2), nb_runner.get_raw_output(2)

    raw = nb_runner.get_output(3)
    data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    assert data["total_overhead"] >= 0.5, (
        f"cash spent at least 0.6s hashing and reported {data['total_overhead']:.2f}s of overhead:\n{raw}"
    )
    assert data["total_compute_time"] < 0.5, (
        f"the hashing was counted as the user's compute: {data['total_compute_time']:.2f}s\n{raw}"
    )
