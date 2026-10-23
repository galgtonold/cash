"""What a fitted model shares with its library only is not searched for.

A fitted ``HistGradientBoostingRegressor`` holds the interpreter's one empty
tuple ``()`` and, through its loss, the ``Interval`` sklearn's link classes
keep as a class attribute. The check for other variables holding the model's
objects climbed their referrers: through the thousands of objects that hold
``()``, and past the link classes into their subclasses, methods and module.
It read the heap seven times over, about 3 s, never ended, and the fit was
refused -- as too costly to check on a slower machine, as shared on a faster
one -- every run. Neither is the notebook's: nothing in ``()`` can change,
and a library's class is found again by its name, as its module is.
"""

from __future__ import annotations

import pytest

import cash.notebook.shared_objects as shared_objects
from tests._cell_driver import run_cash_cell

pytest.importorskip("sklearn")


def test_a_boosting_fit_is_stored_without_searching_what_only_the_library_holds(cash_magics, monkeypatch):
    """The model's loss holds sklearn's ``Interval``, which a variable could
    hold too: its referrers are read, up to the link classes, and the climb
    ends there. ``()`` is never looked for. Every read of what refers to an
    object is watched (`shared_objects._referrers`: `gc.get_referrers`, or on
    3.12.0-3.12.5 with asyncio loaded the heap read `_referrers_of`)."""
    # What is looked for, as (how many, whether ``()`` is among them): a
    # list or a new tuple holding the objects would be one more referrer of
    # them to climb.
    looked_for = []
    empty = tuple()

    def watched(real):
        def read(objs, stop):
            looked_for.append((len(objs), any(obj is empty for obj in objs)))
            return real(objs, stop)

        return read

    climbs = []
    real_climb = shared_objects._variables_holding

    def climb(*args, **kwargs):
        climbs.append(real_climb(*args, **kwargs))
        return climbs[-1]

    monkeypatch.setattr(shared_objects, "_referrers", watched(shared_objects._referrers))
    monkeypatch.setattr(shared_objects, "_referrers_of", watched(shared_objects._referrers_of))
    monkeypatch.setattr(shared_objects, "_variables_holding", climb)
    run_cash_cell(cash_magics, "import numpy as np\nfrom sklearn.ensemble import HistGradientBoostingRegressor")
    run_cash_cell(
        cash_magics,
        "rng = np.random.default_rng(0)\nX = rng.normal(size=(20_000, 4))\ny = X[:, 0] + rng.normal(size=20_000)",
    )
    looked_for.clear()
    climbs.clear()
    run_cash_cell(cash_magics, "m = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)")

    assert not any(has_empty for _n, has_empty in looked_for), "() was looked for up its referrers"
    assert climbs and None not in climbs, "a climb from what the library holds did not end"
    assert looked_for and all(n <= 4 for n, _ in looked_for), looked_for
    row = cash_magics._last_cell_metrics["statements"][-1]
    assert str(row["status"]) == "COMPUTED"
    assert not row.get("uncacheable_reasons"), row.get("uncacheable_reasons")


LIBRARY = (
    "import sys, types\n"
    "lib = types.ModuleType('fakelib')\n"
    "exec('''\n"
    "import time\n"
    "kept = []\n"
    "class Registry:\n"
    "    items = []\n"
    "class Model:\n"
    "    def fit(self, data):\n"
    "        time.sleep(0.3)\n"
    "        self.data = data\n"
    "        KEEP.append(data)\n"
    "        return self\n"
    "''', lib.__dict__)\n"
    "sys.modules['fakelib'] = lib\n"
    "from fakelib import Model, Registry"
)


@pytest.mark.parametrize("keep", ["Registry.items", "kept"], ids=["in_a_library_class_attribute", "in_a_module_global"])
def test_the_notebooks_object_kept_by_a_library_is_never_copied(cash_magics, keep):
    """A library that keeps the notebook's own object in a container of its
    class or its module (``SomeLibClass.registry.append(df)``) shares it with
    the model holding it too. The climb stops at a library's class and
    module, but not then: a stored copy would not be the list the library
    keeps. Both were served as a copy before."""
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, LIBRARY.replace("KEEP", keep))
    for _ in range(2):  # computed, then run again
        run_cash_cell(cash_magics, "m = Model().fit([1, 2, 3])")
        fakelib = ns["sys"].modules["fakelib"]
        kept = fakelib.Registry.items if keep == "Registry.items" else fakelib.kept
        assert ns["m"].data is kept[-1], "m.data is a copy of the list the library keeps"
