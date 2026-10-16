"""A cache hit leaves the global RNG streams where running the notebook would.

The same statement in two cells (``time.sleep(...)``) hits the cache the second
time. Its entry kept where every global stream stood when it was written, and
the hit put them back there: the streams rewound past the draw between the two
cells, and the draw after the hit in its cell repeated that draw. A hit now replays only the streams the
statement moved, so the draws match plain Python.
"""

from __future__ import annotations

import random

import pytest

from tests._nbharness.badge import shows_cached
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.restore]

np = pytest.importorskip("numpy")

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import numpy as np, random, time\nnp.random.seed(0)\nrandom.seed(0)",
    "def draw():\n    return float(np.random.rand()), random.random()",
    "a, ra = draw()",
    SLEEP,
    "b, rb = draw()",
    f"# the same statement again\n{SLEEP}",
    # A draw cash has not seen this cell make yet, so nothing puts the
    # streams back in place before it runs.
    "c, rc = draw()",
]


def _plain() -> list[float]:
    np.random.seed(0)
    random.seed(0)
    draws = [(float(np.random.rand()), random.random()) for _ in range(3)]
    return [value for pair in draws for value in pair]


def test_a_hit_on_a_statement_that_draws_nothing_does_not_repeat_the_draw_before_it(nb_runner):
    nb_runner.create_notebook(CELLS).start_kernel().run_all()
    assert shows_cached(nb_runner.get_output(7)), "the second sleep was not a hit"
    got = [float(nb_runner.peek(name)) for name in ("a", "ra", "b", "rb", "c", "rc")]
    assert got == _plain()
