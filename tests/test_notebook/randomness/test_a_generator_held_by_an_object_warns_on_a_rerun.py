"""A cell re-run on its own that draws from a generator held by an object
cash does not rebuild warns RANDOM-HELD-GENERATOR.

cash follows a generator held in a variable, and one held in an object the
draw changes (``m.rng`` with ``m = M()`` rebuilt). A class attribute read
through an instance, a function's or a module's attribute is not put back
where a run from the top has it, so the re-run draws from where it stands.
"""

import warnings

import pytest

from tests._cell_driver import run_cash_cell

CLASS_HELD = "import numpy as np\nclass C:\n    rng = np.random.default_rng(0)\nc = C()"
METHOD_HELD = (
    "import numpy as np\nclass C:\n    rng = np.random.default_rng(0)\n"
    "    def draw(self):\n        return self.rng.normal()\nc = C()"
)
INSTANCE_HELD = (
    "import numpy as np\nclass M:\n    def __init__(self):\n        self.rng = np.random.default_rng(0)\nm = M()"
)
VARIABLE_HELD = "import numpy as np\nrng = np.random.default_rng(0)"


def _held(caught):
    return [str(w.message) for w in caught if getattr(w.message, "code", None) == "RANDOM-HELD-GENERATOR"]


def _run(cash_magics, cells, indices):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in indices:
            run_cash_cell(cash_magics, cells[i], cells=cells)
    return _held(caught)


@pytest.mark.parametrize(
    "setup, draw",
    [(CLASS_HELD, "c.rng.normal()"), (METHOD_HELD, "c.draw()")],
    ids=["class attribute", "self.rng in a method"],
)
def test_rerunning_a_later_draw_alone_warns_once(cash_magics, setup, draw):
    cells = [setup, f"a = {draw}", f"b = {draw}"]
    assert not _run(cash_magics, cells, [0, 1, 2])
    found = _run(cash_magics, cells, [2, 2])
    assert len(found) == 1
    assert "cell 3" in found[0] and "c.rng" in found[0]


@pytest.mark.parametrize(
    "setup, draw",
    [(INSTANCE_HELD, "m.rng.normal()"), (VARIABLE_HELD, "rng.normal()")],
    ids=["an object rebuilt for the re-run", "a variable"],
)
def test_a_generator_cash_follows_is_quiet(cash_magics, setup, draw):
    """Positive control at the end: the class-held generator in the same
    session warns on the same steps."""
    cells = [setup, f"a = {draw}", f"b = {draw}"]
    assert not _run(cash_magics, cells, [0, 1, 2, 2])
    held = [CLASS_HELD, "a2 = c.rng.normal()", "b2 = c.rng.normal()"]
    _run(cash_magics, held, [0, 1, 2])
    assert _run(cash_magics, held, [2])


def test_running_down_the_notebook_twice_is_quiet(cash_magics):
    cells = [CLASS_HELD, "a = c.rng.normal()", "b = c.rng.normal()"]
    assert not _run(cash_magics, cells, [0, 1, 2, 0, 1, 2])
    assert _run(cash_magics, cells, [2])


def test_the_scan_reads_draws_through_attributes_only():
    from cash.tracking.randomness.detect import attribute_draws

    draws = attribute_draws("x = self.rng.normal() + model.rng.integers(5) + rng.normal() + np.random.rand()")
    chains = {(d.root, d.attrs) for d in draws if d.method is None}
    assert ("self", ("rng",)) in chains and ("model", ("rng",)) in chains
    assert ("np", ("random",)) in chains  # settled on the live object: a module, not a generator
    assert not any(d.root == "rng" and d.method is None for d in draws)
    assert [d.attrs for d in attribute_draws("return self.rng.normal()", "self")] == [("rng",)]
