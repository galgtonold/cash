"""A seed one test sets is not in force when the next test starts.

cash notes each watched ``seed(<value>)`` in a process-wide ledger, which
``reset_session()`` keeps: the stream is still seeded in that process. A warm
kernel is that process for every test on the worker, so a test that ran
``np.random.seed(42)`` left numpy seeded for the next one, and a hidden
unseeded draw there lost its unseeded pill
(randomness/test_rng_badge_indicator.py::test_helper_internal_unseeded_draw_flagged,
which failed behind any seeding test). A fresh kernel starts with no seed in
force, so the warm one forgets it between tests.

The second ``start_kernel()`` is what the next test on the worker does.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

_STILL_SEEDED = (
    "sorted(__import__('cash.tracking.randomness.lineage', fromlist=['x'])"
    ".unseeded_streams({'random', 'numpy.random'}) ^ {'random', 'numpy.random'})"
)


def test_a_seed_set_by_one_test_is_not_in_force_for_the_next(nb_runner):
    nb_runner.create_notebook(
        ["import cash\n%cash_on", "import random\nimport numpy as np\nrandom.seed(1)\nnp.random.seed(42)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(_STILL_SEEDED) == "['numpy.random', 'random']", "control: cash saw both seeds"

    nb_runner.start_kernel()  # what the next test's start does

    assert nb_runner.peek(_STILL_SEEDED) == "[]"
    seeded_state = "__import__('numpy').random.RandomState(42).get_state()[1][:4].tolist()"
    current_state = "__import__('numpy').random.get_state()[1][:4].tolist()"
    assert nb_runner.peek(f"{current_state} != {seeded_state}") == "True", "numpy's stream was left seeded"
