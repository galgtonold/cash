"""Hidden mutation through functools.partial (bound mutable arg) or a function
passed to functools.reduce must reset on isolated re-run (CAS-72)."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_partial_bound_mutable_arg(nb_runner):
    _rerun(nb_runner,
           "from functools import partial\nshared = []\ndef push(lst, v):\n    lst.append(v)\np = partial(push, shared)",
           "p('a')\np('a')\nprint(shared)", "['a', 'a']")


def test_partial_grow_in_loop(nb_runner):
    _rerun(nb_runner,
           "from functools import partial\nbucket = []\ndef add(store, x):\n    store.append(x)",
           "fill = partial(add, bucket)\nfor i in range(3):\n    fill(i)\nprint(bucket)", "[0, 1, 2]")


def test_reduce_side_effecting_reducer(nb_runner):
    _rerun(nb_runner,
           "from functools import reduce\nlog = []\ndef combine(a, b):\n    log.append(b)\n    return a + b",
           "total = reduce(combine, [1, 2, 3], 0)\nprint(total, log)", "6 [1, 2, 3]")


def test_partial_pure_not_over_invalidated(nb_runner):
    # partial over a PURE function must not be flagged / over-invalidated.
    _rerun(nb_runner,
           "from functools import partial\ndef mul(a, b):\n    return a * b\ndouble = partial(mul, 2)",
           "r = double(5)\nprint(r)", "10")


def test_partial_bound_free_var(nb_runner):
    # the partial's target mutates a FREE var (not the bound arg)
    _rerun(nb_runner,
           "from functools import partial\ncounter = [0]\ndef tick(step):\n    counter[0] += step",
           "t = partial(tick, 1)\nt()\nprint(counter[0])", "1")
