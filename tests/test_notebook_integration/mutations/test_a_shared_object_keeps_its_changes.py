"""A change to an object another variable holds survives a second Run All.

A cache hit binds a statement's outputs to deserialised copies. When the object
an output ends up bound to is also held elsewhere -- the list a loop variable
came from, the model a dict was built from, the receiver a call returned -- the
copy is not that object, and a hit loses the change the statement made to it.
Such a statement runs every time instead. Each notebook runs Run All twice in
the same kernel, nothing edited, and must give the plain-Python answer both
times.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = "import time\nimport pandas as pd\ndef slow(x):\n    time.sleep(0.2)\n    return x"

MODEL = (
    "import time\nclass Model:\n    def __init__(self):\n        self.w = 0\n"
    "    def fit(self, k=1):\n        time.sleep(0.3)\n        self.w += k\n        return self"
)

NOTEBOOKS = {
    "frames changed in a loop": (
        [
            SETUP,
            "dfs = [pd.DataFrame({'a': [1.0, 2.0]}), pd.DataFrame({'a': [3.0, 4.0]})]",
            "for d in dfs:\n    d['a'] = slow(d['a'] * 2)",
            "r = sum(float(d['a'].sum()) for d in dfs)",
        ],
        "20.0",
    ),
    "dicts changed in a loop over items()": (
        [
            SETUP,
            "cfgs = {'m1': {'lr': 1}, 'm2': {'lr': 2}}",
            "for name, c in cfgs.items():\n    c['score'] = slow(c['lr'] * 10)",
            "r = [c.get('score') for c in cfgs.values()]",
        ],
        "[10, 20]",
    ),
    "a call that returns its receiver": (
        [MODEL, "m = Model()", "fitted = m.fit(5)", "r = (m.w, m is fitted)"],
        "(5, True)",
    ),
}


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_a_second_run_all_keeps_the_change(nb_runner, name):
    cells, expected = NOTEBOOKS[name]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == expected, "first Run All"
    nb_runner.run_all()
    assert nb_runner.peek("r") == expected, "second Run All"
