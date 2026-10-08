"""``train = data['train']`` then ``train['f'] = ...`` next to a big heap.

The frame is held by ``data``'s dict too, so the statement is stored with
``data`` joined to it. Finding ``data`` walked every object the garbage
collector tracks (``gc.get_referrers``), once per level of containers: with
millions of records elsewhere in the notebook each such statement cost 0.3 s
more, on every Run All. The variables are looked at first now. The answers
must stay those of plain Python, and the time must not follow the heap.
"""

import time

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = "import time\nimport numpy as np, pandas as pd\ndef slow(x):\n    time.sleep(0.25)\n    return x"


def _cells(records):
    return [
        SETUP,
        f"# @cash:no-cache\nrecords = [{{'id': i, 'v': [i]}} for i in range({records})]",
        "data = {'train': pd.DataFrame({'x': [1.0, 2.0]}), 'test': pd.DataFrame({'x': [5.0]})}",
        "train = data['train']",
        "train['f'] = slow(train.x * 2)",
        "train['g'] = train.x * 3",
        "train['h'] = train.x * 4",
        "train['k'] = train.x * 5",
        "r = (list(data['train'].f), list(data['train'].k), data['train'] is train)",
    ]


def test_the_holder_keeps_the_changes_over_two_run_alls(nb_runner):
    nb_runner.create_notebook(_cells(200_000))
    nb_runner.start_kernel()
    for label in ("first", "second"):
        nb_runner.run_all()
        assert nb_runner.peek("r") == "([2.0, 4.0], [5.0, 10.0], True)", f"{label} Run All"


@pytest.mark.perf
def test_a_holder_statement_does_not_pay_for_the_heap(nb_runner):
    """Before: ~0.25 s a statement with three million records (two heap walks
    of ~0.13 s); the statement itself takes milliseconds."""
    nb_runner.create_notebook(_cells(3_000_000))
    nb_runner.start_kernel()
    nb_runner.run_all()
    spent = []
    for cell in (6, 7, 8):
        started = time.perf_counter()
        nb_runner.run_cell(cell)
        spent.append(time.perf_counter() - started)
    assert sorted(spent)[1] < 0.15, [round(s, 3) for s in spent]
    assert nb_runner.peek("data['train'] is train") == "True"
