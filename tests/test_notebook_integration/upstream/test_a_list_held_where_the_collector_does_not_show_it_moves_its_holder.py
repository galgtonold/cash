"""A variable that holds a list through something the garbage collector does
not show it holding moves on when the list is changed in place.

``draw, poke = make(BIG)`` with ``draw`` reaching ``BIG`` through a weak
reference, a pandas object Series or a view of an object array: the share
check reads such an object as one closed value, and asked the collector
what it holds, which shows none of these (a weak reference adds no
reference, numpy does not track its object arrays). No edge joined ``draw``
and ``BIG``, so after ``BIG[0].id = -1000`` was edited, ``y = draw(1)`` was
served from before, and still after a restart. The same for ``H =
weakref.ref(BIG)`` read below as ``H()``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]


HOLDERS = {
    "a pandas object series": ("H = pd.Series([rows, 1], dtype=object)", "H[0]"),
    "a pandas object series of its items": ("H = pd.Series(list(rows), dtype=object)", "H"),
    "a pandas object frame": ("H = pd.DataFrame({'a': [rows, 1]})", "H['a'][0]"),
    "an object array's view": ("H0 = np.empty(2, dtype=object); H0[0] = rows; H = H0[:1]", "H[0]"),
    "a weakref": ("H = weakref.ref(rows)", "H()"),
}

SETUP = (
    "import time, weakref\nimport numpy as np, pandas as pd\nclass Row:\n    def __init__(self, i):\n"
    "        self.id = i\nclass Bag(list):\n    pass\nBIG = Bag(Row(i) for i in range(2000))\n"
)


def _edit_and_run(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for edit in (-1000, -2000, -3000):
        nb_runner.set_cell_source(4, f"BIG[0].id = {edit}")
        if edit == -3000:
            nb_runner.restart()
        nb_runner.run_all()
        assert nb_runner.peek("y") == str(sum(range(1, 2000)) + edit), nb_runner.get_raw_output(5)


@pytest.mark.parametrize("rebind", [False, True], ids=["alone", "beside a closure rebinding rows"])
@pytest.mark.parametrize("holder", sorted(HOLDERS))
def test_a_closure_holding_the_list_where_the_collector_cannot_see_moves_with_it(nb_runner, holder, rebind):
    """``draw`` holds ``BIG`` only through *holder*: alone, no edge was made
    at all, and ``y`` was served from before the edit. Beside ``rebind``,
    whose cell holds ``BIG`` itself, ``BIG`` is the statement's anchor, and
    finding it inside *holder* must not refuse what the anchor allows, and
    ``draw`` shares ``BIG``'s rows with ``rebind`` through *holder*, which
    the outputs' link to ``BIG`` must see, or it named ``rebind`` alone."""
    hold, read = HOLDERS[holder]
    extra = "    def rebind(r):\n        nonlocal rows\n        rows = r\n" if rebind else ""
    names = "draw, poke, rebind" if rebind else "draw, poke"
    _edit_and_run(
        nb_runner,
        [
            "import cash\n%cash_on",
            SETUP + f"def make(rows):\n    {hold}\n    def draw(k):\n"
            f"        time.sleep(0.2)\n        return sum(r.id for r in {read})\n"
            f"    def poke():\n        return len(str(H))\n{extra}    return {names}\n{names} = make(BIG)",
            "poke()",
            "BIG[0].id = -1000",
            "y = draw(1)",
        ],
    )


@pytest.mark.parametrize("holder", sorted(HOLDERS))
def test_a_variable_holding_the_list_where_the_collector_cannot_see_moves_with_it(nb_runner, holder):
    """``H`` holds ``BIG`` only through *holder*: unlinked, ``y`` reading
    ``BIG`` through ``H`` was served from before the edit."""
    hold, read = HOLDERS[holder]
    _edit_and_run(
        nb_runner,
        [
            "import cash\n%cash_on",
            SETUP + "rows = BIG\n" + hold + "\ndel rows",
            "time.sleep(0.2)",
            "BIG[0].id = -1000",
            f"y = sum(r.id for r in {read})",
        ],
    )
