"""A function an object holds in a ``dict`` subclass set as its ``__dict__``
is followed like one held in a plain attribute.

``h = H(f)`` with ``self.__dict__ = Attrs(fn=g)``, and ``r = h.fn(3)`` with
``f`` reading ``K = 2`` from a cell above. A ``K = 5`` cell added below and
run, then ``r = h.fn(3)`` run again: the walk for the functions ``h`` holds
read the dict through the subclass's own methods and lost ``f``, so the
statement re-ran on ``K = 5`` and gave 15, where a plain attribute brings
``K = 2`` back and serves 6.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ATTRS = "class Attrs(dict):\n    def values(self):\n        return list(super().values())\n"
HOLDERS = {
    "dict subclass": "class H:\n    def __init__(self, g):\n        self.__dict__ = Attrs(fn=g)\n",
    "plain attribute": "class H:\n    def __init__(self, g):\n        self.fn = g\n",
}


@pytest.mark.parametrize("holder", HOLDERS.values(), ids=HOLDERS.keys())
def test_a_cell_added_below_does_not_reach_the_held_function(nb_runner, holder):
    nb_runner.create_notebook(
        [
            "%cash_badge print\nK = 2",
            "import time\ndef f(x):\n    time.sleep(0.3)\n    return x * K",
            ATTRS + holder,
            "h = H(f)",
            "r = h.fn(3)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == "6"

    nb_runner.add_cell("K = 5", save=True)
    nb_runner.run_cell(6)
    nb_runner.run_cell(5)

    raw = nb_runner.get_raw_output(5)
    assert (nb_runner.peek("r"), nb_runner.peek("K")) == ("6", "2"), raw
    assert "CACHED: r = h.fn(3)" in raw, raw
