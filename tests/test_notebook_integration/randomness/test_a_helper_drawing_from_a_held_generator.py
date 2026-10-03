"""A cached helper call that draws from a generator held in a variable.

Seen in a notebook::

    rng = np.random.default_rng(42)
    def boot(x): ...rng.integers(...)...
    res = {g: boot(x) for g, x in groups.items()}   # 10 groups, then 60

Each ``boot(x)`` was cached on its own. On the 60-group run the first ten
calls hit, the hits did not advance ``rng``, and the other fifty drew from
the wrong place in the stream: most results differed from a plain run, and
those wrong values were stored.

A call that consumes a module-level RNG was already refused; one that
consumes a generator the helper reaches through its globals was not seen.
The oracle is the same cell run without cash.
"""

import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

CELL = """import numpy as np, json
groups = {f"g{i:03d}": np.arange(200.0) + i for i in range(N)}
rng = np.random.default_rng(42)
def boot(x, n=300):
    out = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        out.append(x[i].mean())
    return float(np.percentile(out, 90))
res = {g: boot(x) for g, x in groups.items()}
print("RESULT", json.dumps(res))"""


def _plain(n):
    ns = {}
    exec(f"N = {n}\n" + CELL.replace('print("RESULT", json.dumps(res))', ""), ns)
    return ns["res"]


def _result(nb_runner):
    out = nb_runner.get_output(3)
    return json.loads(next(line for line in out.splitlines() if line.startswith("RESULT"))[len("RESULT ") :])


def test_more_groups_after_fewer_match_a_plain_run(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", "N = 10", CELL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _result(nb_runner) == _plain(10)

    nb_runner.set_cell_source(2, "N = 60")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert _result(nb_runner) == _plain(60)
