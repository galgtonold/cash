"""A later draw re-run alone from a generator held by an object.

cash follows a generator held in a variable and one an object holds that
the re-run rebuilds. A class attribute read through an instance (or a
module's, or a function's) is not put back where a run from the top has it:
the re-run draws from where it stands, and cash says so
(RANDOM-HELD-GENERATOR). A run from the top is quiet.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

CLASS_HELD = "import numpy as np\nclass C:\n    rng = np.random.default_rng(0)\nc = C()"
INSTANCE_HELD = (
    "import numpy as np\nclass M:\n    def __init__(self):\n        self.rng = np.random.default_rng(0)\nm = M()"
)


def _notebook(setup, receiver):
    return [setup, f"a = {receiver}.rng.normal()\nprint(a)", f"b = {receiver}.rng.normal()\nprint(b)"]


def test_the_rerun_alone_warns(nb_runner):
    nb_runner.create_notebook(_notebook(CLASS_HELD, "c"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert "RANDOM-HELD-GENERATOR" not in nb_runner.get_raw_output(2) + nb_runner.get_raw_output(3)
    nb_runner.run_cell(3)
    raw = nb_runner.get_raw_output(3)
    assert "RANDOM-HELD-GENERATOR" in raw
    assert "cell 3" in raw and "c.rng" in raw


def test_a_generator_on_an_object_the_rerun_rebuilds_is_quiet(nb_runner):
    """It is followed: the re-run draws where a run from the top does. The
    class-held generator below it, re-run the same way, is the positive control."""
    nb_runner.create_notebook(_notebook(INSTANCE_HELD, "m") + _notebook(CLASS_HELD, "c"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    top_to_bottom = nb_runner.peek("b")
    nb_runner.run_cell(3)
    assert "RANDOM-HELD-GENERATOR" not in nb_runner.get_raw_output(3)
    assert nb_runner.peek("b") == top_to_bottom
    nb_runner.run_cell(6)
    assert "RANDOM-HELD-GENERATOR" in nb_runner.get_raw_output(6)
