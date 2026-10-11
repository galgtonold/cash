"""A loop that changes the items it walks changes the list they are in.

``for r in records: r['tax'] = ...`` changes ``records``. After the loop is
edited and the notebook runs again, the cell reading ``records`` must not be
served from the cache with the old items. Each case is compared with plain
Python.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = (
    "import time\nimport numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "def slow(v):\n    time.sleep(0.2)\n    return v"
)

CASES = {
    "dicts": (
        "records = [{'p': 10}, {'p': 20}]",
        "for r in records:\n    r['tax'] = r['p'] * 2",
        "for r in records:\n    r['vat'] = r['p'] * 2",
        "r = slow(sorted(records[0]))\nprint(r)",
        "['p', 'tax']",
        "['p', 'vat']",
    ),
    "lists": (
        "rows = [[1], [2]]",
        "for r in rows:\n    r.append(5)",
        "for r in rows:\n    r.append(6)",
        "r = slow(repr(rows))\nprint(r)",
        "[[1, 5], [2, 5]]",
        "[[1, 6], [2, 6]]",
    ),
    "arrays": (
        "arrs = [np.zeros(2), np.ones(2)]",
        "for a in arrs:\n    a += 3",
        "for a in arrs:\n    a += 4",
        "r = slow(float(sum(x.sum() for x in arrs)))\nprint(r)",
        "14.0",
        "18.0",
    ),
    "axes": (
        "fig, axes = plt.subplots(1, 2)",
        "for ax in axes:\n    ax.plot([1, 2])",
        "for ax in axes:\n    ax.plot([1, 2])\n    ax.plot([2, 1])",
        "r = slow(len(axes[0].lines))\nprint(r)",
        "1",
        "2",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_an_edited_loop_reaches_the_reader(nb_runner, name):
    make, loop, edited, read, before, after = CASES[name]
    nb_runner.create_notebook([SETUP, make, loop, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == before
    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == after
