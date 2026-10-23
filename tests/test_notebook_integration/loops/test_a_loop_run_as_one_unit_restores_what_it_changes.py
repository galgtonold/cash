"""A loop run as one unit stores the objects it changes in place.

``for f in files: ...; pbar.update()`` changes ``pbar`` and binds nothing
that holds it. The loop ran again on every Run All and after every restart,
as one changing ``st.push(v)`` did. It now stores what it changed with its
outputs, and a hit restores it: the objects, and every name or container that
holds them, are what plain Python gives.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

P = (
    "import time\nclass P:\n    def __init__(self):\n        self.n = 0\n        self.seen = []\n"
    "    def update(self, i=0):\n        self.n += 1\n        self.seen.append(i)\n"
)
LOOP = "for i in range({n}):\n    time.sleep(0.0003)\n    p.update(i)"
STATUS = "[s['status'] for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"


CASES = {
    "another name": (P + "p = P()\nq = p", "print(p.n, q.n, p is q, len(q.seen))", "3000 3000 True 3000"),
    "a container": (P + "p = P()\nholder = {'a': [p]}", "print(holder['a'][0].n, holder['a'][0] is p)", "3000 True"),
    "a later cell": (P + "p = P()", "print(p.n, sum(p.seen))", "3000 4498500"),
    "some passes": (
        P + "p = P()",
        "print(p.n, p.seen[-1])",
        "429 2996",
    ),
}
LOOPS = {"some passes": "for i in range(3000):\n    time.sleep(0.0003)\n    if i % 7 == 0:\n        p.update(i)"}


def _restart_and_run_one_by_one(nb_runner):
    """After a restart, run the cells one by one: the loop cell's status is
    read before the cell after it runs."""
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    statuses = nb_runner.peek(STATUS)
    nb_runner.run_cell(3)
    return statuses


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", list(CASES))
def test_the_loop_is_restored_and_what_holds_the_object_sees_it(nb_runner, case):
    make, read, expected = CASES[case]
    nb_runner.create_notebook([make, LOOPS.get(case, LOOP.format(n=3000)), read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == expected
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == expected
    assert "RESTORED" in _restart_and_run_one_by_one(nb_runner)
    assert nb_runner.get_output(3).strip() == expected


@pytest.mark.fresh_kernel
def test_a_longer_range_after_a_hit_runs_the_loop_again(nb_runner):
    nb_runner.create_notebook([P + "p = P()", LOOP.format(n=3000), "print(p.n)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3000"
    nb_runner.set_cell_source(2, LOOP.format(n=3500))
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3500"
    nb_runner.set_cell_source(2, LOOP.format(n=3000))
    assert "RESTORED" in _restart_and_run_one_by_one(nb_runner)
    assert nb_runner.get_output(3).strip() == "3000"


@pytest.mark.fresh_kernel
def test_an_object_the_store_cannot_keep_runs_the_loop(nb_runner):
    """A lock in the object: not stored, the loop runs, as before."""
    make = P + "import threading\np = P()\np.lock = threading.Lock()"
    nb_runner.create_notebook([make, LOOP.format(n=3000), "print(p.n)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3000"
    assert "RESTORED" not in _restart_and_run_one_by_one(nb_runner)
    assert nb_runner.get_output(3).strip() == "3000"
