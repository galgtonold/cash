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


# The walk is a method named like a file listing, a function of the
# user's hands the items over, or each item is handed to a function of the
# user's. The edit is to the loop or to a cell above it.
WALKS = {
    "a tree's walk()": (
        [
            "import time\nclass Tree:\n    def __init__(self, vals):\n        self.nodes = [{'v': v} for v in vals]\n"
            "    def walk(self):\n        yield from self.nodes\n"
            "def summarize(t):\n    time.sleep(0.2)\n    return [n['v'] * 100 + n['tax'] for n in t.nodes]\n"
            "tree = Tree([1, 2, 3])",
            "for n in tree.walk():\n    n['tax'] = n['v'] * 2",
            "s = summarize(tree)\nprint(s)",
        ],
        (2, "for n in tree.walk():\n    n['tax'] = n['v'] * 3"),
        3,
        "[102, 204, 306]",
        "[103, 206, 309]",
    ),
    "records a function returns": (
        [
            "import time\nrecords = [{'v': v, 'on': v % 2 == 1} for v in range(1, 7)]\n"
            "def active():\n    return [r for r in records if r['on']]\n"
            "def summarize(rs):\n    time.sleep(0.2)\n    return [r.get('tax') for r in rs]\nrate = 2",
            "for r in active():\n    r['tax'] = r['v'] * rate",
            "s = summarize(records)\nprint(s)",
        ],
        (
            1,
            "import time\nrecords = [{'v': v, 'on': v % 2 == 1} for v in range(1, 7)]\n"
            "def active():\n    return [r for r in records if r['on']]\n"
            "def summarize(rs):\n    time.sleep(0.2)\n    return [r.get('tax') for r in rs]\nrate = 3",
        ),
        3,
        "[2, None, 6, None, 10, None]",
        "[3, None, 9, None, 15, None]",
    ),
    "each item handed to a function": (
        [
            "import time\ndef summarize(recs):\n    time.sleep(0.2)\n    return sorted(set(k for r in recs for k in r))",
            "def add_tax(r):\n    r['tax'] = r['p'] * 0.2",
            "records = [{'p': 1}, {'p': 2}, {'p': 3}]",
            "for r in records:\n    add_tax(r)",
            "fields = summarize(records)\nprint(fields)",
        ],
        (2, "def add_tax(r):\n    r['vat'] = r['p'] * 0.2"),
        5,
        "['p', 'tax']",
        "['p', 'vat']",
    ),
    "each frame handed to a function": (
        [
            "import time\nimport pandas as pd\ndef summarize(frames):\n    time.sleep(0.2)\n"
            "    return [list(f.columns) for f in frames]",
            "def add_features(df):\n    df['tax'] = df['p'] * 2",
            "frames = [pd.DataFrame({'p': [1, 2]}), pd.DataFrame({'p': [3]})]",
            "for df in frames:\n    add_features(df)",
            "cols = summarize(frames)\nprint(cols)",
        ],
        (2, "def add_features(df):\n    df['vat'] = df['p'] * 2"),
        5,
        "[['p', 'tax'], ['p', 'tax']]",
        "[['p', 'vat'], ['p', 'vat']]",
    ),
}


@pytest.mark.parametrize("name", list(WALKS))
def test_an_edit_reaches_the_reader_of_what_the_loop_walks(nb_runner, name):
    cells, (edit_at, edited), read_at, before, after = WALKS[name]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(read_at).strip() == before
    nb_runner.set_cell_source(edit_at, edited)
    nb_runner.run_all()
    assert nb_runner.get_output(read_at).strip() == after
