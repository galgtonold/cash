"""A long loop run as one unit evaluates its header once, as plain Python does.

The handler evaluates the header to size the loop. The unit used to run the
loop from source, which evaluated the header a second time, so a header that
draws or consumes gave the loop a second, different sequence, with a clean
badge on the first Run All:

* ``for i in np.random.permutation(300):`` iterated the SECOND permutation and
  moved the generator twice, so every later draw differed too (likewise
  ``random.sample(items, 300)`` and ``df.sample(n=300).index``);
* ``sorted(inbox.drain())``, ``list(b.gen)`` and ``list(gens['a'])`` ran zero
  times: the second evaluation found the queue or generator empty;
* ``list(map(f, range(300)))`` called ``f`` 600 times;
* sizing ``enumerate(loader.batch)`` read the property once more, and the
  loop iterated a later batch.

The unit now iterates the value the handler evaluated. Each expectation below
is what plain Python prints for the same cells.
"""

import random

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(180)]

CASH_ON = "import cash\n%cash_on"


def _plain(*cells: str) -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


def _run(nb_runner, *cells: str) -> str:
    nb_runner.create_notebook([CASH_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return nb_runner.get_output(len(cells) + 1).strip()


def test_a_random_permutation_is_drawn_once(nb_runner):
    pytest.importorskip("numpy")
    seed = "import numpy as np\nnp.random.seed(0)"
    loop = "order = []\nfor i in np.random.permutation(300):\n    order.append(int(i))\nlater = int(np.random.randint(1000))"
    ns = _plain(seed, loop)
    expected = f"{ns['order'][:5]} {ns['later']}"
    out = _run(nb_runner, seed, loop + "\nprint(order[:5], later)")
    assert out == expected, f"cash iterated another permutation, or drew twice: {out!r}, plain {expected!r}"
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == expected, "a second Run All changed the draw"


def test_a_random_sample_is_drawn_once(nb_runner):
    setup = "import random\nrandom.seed(1)\nitems = list(range(1000))"
    loop = "picked = []\nfor x in random.sample(items, 300):\n    picked.append(x)"
    expected = str(_plain(setup, loop)["picked"][:5])
    random.seed()
    assert _run(nb_runner, setup, loop + "\nprint(picked[:5])") == expected


def test_a_frame_sample_is_drawn_once(nb_runner):
    pytest.importorskip("pandas")
    setup = "import numpy as np, pandas as pd\nnp.random.seed(0)\ndf = pd.DataFrame({'x': range(1000)})"
    loop = "picked = []\nfor idx in df.sample(n=300).index:\n    picked.append(int(idx))"
    expected = str(_plain(setup, loop)["picked"][:5])
    assert _run(nb_runner, setup, loop + "\nprint(picked[:5])") == expected


@pytest.mark.parametrize(
    "setup,header",
    [
        (
            "class Inbox:\n    def __init__(self): self.items = list(range(300))\n"
            "    def drain(self):\n        out, self.items = self.items, []\n        return out\ninbox = Inbox()",
            "sorted(inbox.drain())",
        ),
        ("class Box: pass\nb = Box()\nb.gen = (i for i in range(300))", "list(b.gen)"),
        ("gens = {'a': (i for i in range(300))}", "list(gens['a'])"),
    ],
    ids=["a drained queue", "a generator on an attribute", "a generator in a dict"],
)
def test_a_header_that_consumes_is_consumed_once(nb_runner, setup, header):
    out = _run(nb_runner, setup, f"total = 0\nfor v in {header}:\n    total += v\nprint(total)")
    assert out == "44850", f"the loop ran over a second, empty evaluation of {header}: {out!r}"


def test_a_function_mapped_over_a_range_is_called_once_per_item(nb_runner):
    out = _run(
        nb_runner,
        "calls = []\ndef f(x):\n    calls.append(x)\n    return x * 2",
        "tot = 0\nfor v in list(map(f, range(300))):\n    tot += v\nprint(tot, 'calls', len(calls))",
    )
    assert out == "89700 calls 300", out


def test_a_property_in_the_header_is_read_once(nb_runner):
    loader = (
        "class Loader:\n    def __init__(self):\n        self.epoch = 0\n    @property\n    def batch(self):\n"
        "        self.epoch += 1\n        return list(range(self.epoch * 1000, self.epoch * 1000 + 300))\n"
        "loader = Loader()"
    )
    out = _run(nb_runner, loader, "tot = 0\nfor i, x in enumerate(loader.batch):\n    tot += x\nprint(tot, 'epoch', loader.epoch)")
    assert out == "344850 epoch 1", out
