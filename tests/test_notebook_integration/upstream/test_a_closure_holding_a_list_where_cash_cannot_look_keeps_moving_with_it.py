"""A closure that holds a list through something cash does not open keeps
moving with the list after a call that rebinds nothing.

``draw, poke = make(BIG)``, with ``draw`` reaching ``BIG`` through a pandas
object Series, an object array's view or a weakref: after ``poke()`` the
edge between ``draw`` and ``BIG`` was asked again (`holds_part_of`), which
does not look inside those holders, read as holding no part of ``BIG``, and
was dropped. An edit of the change to ``BIG`` below then served ``y`` from
before, and still after a restart. An edge is dropped now only when all of
both sides can be read.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]


HOLDERS = {
    "a pandas object series": ("H = pd.Series([rows, 1], dtype=object)", "H[0]", "len(H)"),
    "an object array's view": ("H0 = np.empty(2, dtype=object); H0[0] = rows; H = H0[:1]", "H[0]", "len(H)"),
    "a weakref": ("H = weakref.ref(rows)", "H()", "H is None"),
}


@pytest.mark.parametrize("holder", sorted(HOLDERS))
def test_a_closure_holding_the_list_where_cash_cannot_look_keeps_moving_with_it(nb_runner, holder):
    """``poke()`` rebinds nothing (``rebind`` would); ``draw`` holds ``BIG`` through something
    `holds_part_of` does not open, which read as holding none of it, and the
    edge was dropped: after an edit of the change below, ``y`` was served
    from before, and still after a restart."""
    hold, read, touch = HOLDERS[holder]
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import time, weakref\nimport numpy as np, pandas as pd\nclass Row:\n    def __init__(self, i):\n"
            "        self.id = i\nclass Bag(list):\n    pass\nBIG = Bag(Row(i) for i in range(2000))\n"
            f"def make(rows):\n    {hold}\n    def draw(k):\n        time.sleep(0.2)\n"
            f"        return sum(r.id for r in {read})\n    def poke():\n        return {touch}\n"
            "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, poke, rebind\n"
            "draw, poke, rebind = make(BIG)",
            "poke()",
            "BIG[0].id = -1000",
            "y = draw(1)",
        ]
    )
    nb_runner.start_kernel()
    for edit in (-1000, -2000, -3000):
        nb_runner.set_cell_source(4, f"BIG[0].id = {edit}")
        if edit == -3000:
            nb_runner.restart()
        nb_runner.run_all()
        assert nb_runner.peek("y") == str(sum(range(1, 2000)) + edit), nb_runner.get_raw_output(5)
