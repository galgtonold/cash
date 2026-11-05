"""What the upstream check runs again to put a random stream in place.

Before a draw the plan re-runs, the global stream must stand where a run from
the top has it. The planner sets it by running again only a bare seed
(``np.random.seed(1)``) at its own place in the run order, never a whole cell
and never a statement that does more than seed: a draw above with other work
is put back from a recorded position, or not at all, with a warning.

The planner is a function of the notebook's cells, the planned statements,
the live namespace and what the kernel recorded, so these tests drive it
without a kernel.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

from cash.notebook.upstream import UpstreamChecker
from cash.notebook.upstream._types import InputHashes, TraceEntry
from cash.tracking.randomness import rng_virtual_var

NP = rng_virtual_var("numpy.random")


def _checker(live: dict | None = None) -> UpstreamChecker:
    shell = MagicMock()
    shell.user_ns = dict(live or {})
    return UpstreamChecker(shell=shell)


def _plan(checker, cells, statements, trace=None, restored=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, added = checker.rng.with_rng_chain("print(1)", cells, len(cells), statements, trace, restored)
    codes = [getattr(w.message, "code", None) for w in caught]
    return out, added, codes


SAME_SEEDS = [
    "import numpy as np",
    "np.random.seed(1)\na = np.random.rand(3)",
    "np.random.seed(1)\nb = np.random.rand(2)",
    "np.random.seed(1)\nc = np.random.rand(4)",
]


def test_each_seed_is_run_at_its_own_place():
    """Three cells open with the same seed: each draw gets the one above it."""
    out, added, _ = _plan(_checker({"np": object()}), SAME_SEEDS, ["b = np.random.rand(2)", "c = np.random.rand(4)"])
    assert out == ["np.random.seed(1)", "b = np.random.rand(2)", "np.random.seed(1)", "c = np.random.rand(4)"]
    assert added == {"np.random.seed(1)"}


def test_no_cell_is_run_whole():
    """Only the seed statement goes in, not the cell around it."""
    cells = ["import numpy as np", "np.random.seed(1)\nw = np.random.rand(3)\nmodel = 'first'", "z = np.random.rand(2)"]
    out, _, _ = _plan(_checker({"np": object()}), cells, ["z = np.random.rand(2)"])
    assert out == ["z = np.random.rand(2)"], "the draw above, not the seed, moved the stream last"
    out, _, _ = _plan(_checker({"np": object()}), cells, ["w = np.random.rand(3)"])
    assert out == ["np.random.seed(1)", "w = np.random.rand(3)"]


SHUFFLE = [
    "import numpy as np\nimport pandas as pd",
    "np.random.seed(0)",
    "df = df.reindex(np.random.permutation(df.index))",
    "z = np.random.rand(k)",
]


def test_a_draw_that_does_more_is_never_run_again():
    """Nothing recorded the stream after the shuffle: the draw runs from
    where the stream stands, and cash says so."""
    out, added, codes = _plan(_checker({"np": object(), "df": object(), "k": 3}), SHUFFLE, ["z = np.random.rand(k)"])
    assert out == ["z = np.random.rand(k)"]
    assert not added
    assert "RANDOM-STREAM-NOT-RESTORED" in codes


def test_a_recorded_position_is_put_back_instead():
    """The kernel recorded the stream under the lineage the draw reads: it
    goes back there just before the draw, and nothing runs again."""
    checker = _checker({"np": object(), "df": object(), "k": 3})
    state = {"numpy.random": ("MT19937", "recorded")}
    checker.tracking_state.rng_states_by_lineage[(NP, "after-shuffle")] = state
    trace = [
        TraceEntry(
            "z = np.random.rand(k)", {"z"}, {"np", "k"}, InputHashes({}, None, {NP: "after-shuffle"}), {}, False, cell=3
        )
    ]
    out, _, codes = _plan(checker, SHUFFLE, ["z = np.random.rand(k)"], trace)
    assert out == ["z = np.random.rand(k)"]
    assert "RANDOM-STREAM-NOT-RESTORED" not in codes
    assert checker.rng._before == {0: state}


def test_a_restored_draw_puts_the_stream_where_its_run_left_it():
    checker = _checker({"np": object(), "df": object(), "k": 3})
    state = {"numpy.random": ("MT19937", "entry")}
    restored = {"df = df.reindex(np.random.permutation(df.index))": state}
    out, _, codes = _plan(checker, SHUFFLE, ["z = np.random.rand(k)"], restored=restored)
    assert out == ["z = np.random.rand(k)"]
    assert "RANDOM-STREAM-NOT-RESTORED" not in codes
    assert checker.rng._before == {0: state}


def test_a_seed_brings_the_import_it_reads():
    """After a restart ``np`` is not bound: the import above the seed runs first."""
    cells = ["import numpy as np\nnp.random.seed(0)", "x = float(np.random.rand()) + 1"]
    out, _, _ = _plan(_checker({}), cells, ["x = float(np.random.rand()) + 1"])
    assert out == ["import numpy as np", "np.random.seed(0)", "x = float(np.random.rand()) + 1"]
    out, _, _ = _plan(_checker({}), cells, ["import numpy as np", "x = float(np.random.rand()) + 1"])
    assert out == ["import numpy as np", "np.random.seed(0)", "x = float(np.random.rand()) + 1"]


def test_a_draw_nothing_seeds_above_is_left_alone():
    """Unseeded where it stands: no position a run from the top fixes."""
    cells = ["import numpy as np", "a = np.random.rand(2)", "b = np.random.rand(2)"]
    trace = [TraceEntry("b = np.random.rand(2)", {"b"}, {"np"}, InputHashes({}, None, {NP: None}), {}, False, cell=2)]
    out, _, codes = _plan(_checker({"np": object()}), cells, ["b = np.random.rand(2)"], trace)
    assert out == ["b = np.random.rand(2)"]
    assert "RANDOM-STREAM-NOT-RESTORED" not in codes


def test_no_draw_in_the_plan_changes_nothing():
    out, added, _ = _plan(_checker({"np": object()}), SAME_SEEDS, ["total = 1 + 2"])
    assert out == ["total = 1 + 2"]
    assert not added


CHAIN = ["import numpy as np", "np.random.seed(0)\nn = 3", "a = np.random.rand(n)", "b = np.random.rand(2)"]


def test_a_draw_that_only_binds_its_names_runs_again_from_its_own_place():
    """``a = np.random.rand(n)`` binds ``a`` and nothing rebinds it or ``n``
    below: run again (after the seed above it), it holds what a run gives."""
    import numpy

    out, _, codes = _plan(_checker({"np": numpy, "n": 3}), CHAIN, ["b = np.random.rand(2)"])
    assert out == ["np.random.seed(0)", "a = np.random.rand(n)", "b = np.random.rand(2)"]
    assert "RANDOM-STREAM-NOT-RESTORED" not in codes


def test_a_draw_whose_names_change_below_is_not_run_again():
    import numpy

    cells = [*CHAIN[:3], "a = a * 2", "b = np.random.rand(2)"]
    out, _, codes = _plan(_checker({"np": numpy, "n": 3}), cells, ["b = np.random.rand(2)"])
    assert out == ["b = np.random.rand(2)"]
    assert "RANDOM-STREAM-NOT-RESTORED" in codes
    cells = [*CHAIN[:3], "n = 4", "b = np.random.rand(2)"]
    out, _, codes = _plan(_checker({"np": numpy, "n": 4}), cells, ["b = np.random.rand(2)"])
    assert out == ["b = np.random.rand(2)"]
