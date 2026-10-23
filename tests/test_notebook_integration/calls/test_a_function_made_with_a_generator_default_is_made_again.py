"""``make()`` returns a function whose default is a new generator
(``def h(g=gen())``). Plain Python makes a new one on every Run All, so
``x = h()`` and ``y = h()`` read 0 and 1 each time.

Slow enough to store, ``make()`` was served from the call cache: the
function it handed back still held the generator the last Run All had drawn
from, and ``x, y`` went on 2, 3 and then 4, 5.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

CELLS = [
    "import time\n"
    "def gen():\n    i = 0\n    while True:\n        yield i\n        i += 1\n"
    "def make():\n    time.sleep(0.05)\n    def h(g=gen()):\n        return next(g)\n    return h",
    "h = make()",
    "x = h()",
    "y = h()",
]


def test_each_run_all_draws_from_a_new_generator(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    for _ in range(3):
        nb_runner.run_all()
        assert (nb_runner.peek("x"), nb_runner.peek("y")) == ("0", "1")
