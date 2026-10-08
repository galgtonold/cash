"""A seed set through a notebook helper counts as a seed.

``def set_seed(s): random.seed(s); np.random.seed(s)`` then ``set_seed(42)``
is the usual way to seed a notebook. Only a statement that spelled the seed
call wrote the RNG variable a draw's key reads, so changing it to
``set_seed(43)`` left the draws below keyed as before, and Run All served the
seed-42 numbers.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from cash.notebook.callee_reach import helper_seeded_modules
from cash.tracking.randomness import rng_virtual_var
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

NP = rng_virtual_var("numpy.random")
SETUP = (
    "import random, time\n"
    "import numpy as np\n"
    "def slow(v):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return v\n"
    "def set_seed(s):\n"
    "    random.seed(s)\n"
    "    np.random.seed(s)"
)
DRAW = "X = slow(np.random.rand(2))"


@pytest.fixture
def run(cash_magics, mock_shell, clean_backend):
    def _run(*cells):
        statuses = []
        for code in cells:
            run_cash_cell(cash_magics, code)
            statuses += [str(m.get("status")) for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
        return statuses

    _run.ns = mock_shell.user_ns
    _run.lineage = cash_magics._cell_executor.tracking_state.variable_lineage
    return _run


def _plain(seed):
    np.random.seed(seed)
    return np.random.rand(2).tolist()


def test_a_new_seed_through_the_helper_gives_new_numbers(run):
    run(SETUP, "set_seed(42)", DRAW)
    assert run.ns["X"].tolist() == _plain(42)
    run("set_seed(43)", DRAW)
    assert run.ns["X"].tolist() == _plain(43)


def test_the_same_seed_through_the_helper_is_served_from_the_cache(run):
    run(SETUP, "set_seed(42)", DRAW)
    assert run("set_seed(42)", DRAW)[-1] == "RESTORED"
    assert run.ns["X"].tolist() == _plain(42)


def test_the_helper_call_writes_the_rng_variables(run):
    run(SETUP, "set_seed(42)")
    at_42 = run.lineage[NP]
    assert rng_virtual_var("random") in run.lineage
    run("set_seed(43)")
    assert run.lineage[NP] != at_42


def test_a_seed_two_helpers_deep_counts(run):
    run(SETUP, "def setup(s):\n    set_seed(s)", "setup(42)", DRAW)
    run("setup(43)", DRAW)
    assert run.ns["X"].tolist() == _plain(43)


def test_the_random_module_follows_the_helper_too(run):
    draw = "r = slow(random.random())"
    run(SETUP, "set_seed(42)", draw)
    run("set_seed(43)", draw)
    random.seed(43)
    assert run.ns["r"] == random.random()


def test_only_a_call_of_a_notebook_function_that_seeds_is_followed():
    ns: dict = {}
    exec("import numpy as np\ndef other(s):\n    return s\n", ns)
    assert helper_seeded_modules("other(42)", ns) == frozenset()
    assert helper_seeded_modules("len([1])", ns) == frozenset()
    assert helper_seeded_modules("def set_seed(s):\n    np.random.seed(s)", ns) == frozenset()


def test_a_def_the_simulation_saw_is_read_when_the_kernel_has_none():
    """After a restart the simulation reaches ``set_seed(43)`` before the
    kernel ran the ``def`` again."""
    sources = {"set_seed": "def set_seed(s):\n    random.seed(s)\n    np.random.seed(s)"}
    assert helper_seeded_modules("set_seed(43)", {}, sources.get) == {"random", "numpy.random"}
