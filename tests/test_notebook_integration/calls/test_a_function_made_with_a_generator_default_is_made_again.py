"""``make()`` returns a function whose default is a new generator
(``def h(g=gen())``). Plain Python makes a new one on every Run All, so
``x = h()`` and ``y = h()`` read 0 and 1 each time.

Slow enough to store, ``make()`` was served from the call cache: the
function it handed back still held the generator the last Run All had drawn
from, and ``x, y`` went on 2, 3 and then 4, 5. It is now not stored, as a
counter a factory returns is not: the statement shows what that one shows,
computed again with no call served.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

CELLS = [
    "import time\n"
    "def gen():\n    i = 0\n    while True:\n        yield i\n        i += 1\n"
    "def make():\n    time.sleep(0.05)\n    def h(g=gen()):\n        return next(g)\n    return h\n"
    "def counter():\n    time.sleep(0.05)\n    n = [0]\n    def c():\n        n[0] += 1\n        return n[0]\n    return c",
    "h = make()",
    "c = counter()",
    "x = h()",
    "y = h()",
]

SHOWN = (
    "[(str(s.get('status')).split('.')[-1], (s.get('calls') or {}).get('restored'), (s.get('calls') or {}).get('computed'))"
    " for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics.get('statements', [])]"
)


def test_each_run_all_draws_from_a_new_generator(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    for run in range(3):
        shown = {}
        for cell in range(1, len(CELLS) + 1):
            nb_runner.run_cell(cell)
            if cell in (2, 3):
                shown[cell] = nb_runner.peek(SHOWN)
        assert (nb_runner.peek("x"), nb_runner.peek("y")) == ("0", "1")
        if run:
            assert shown[2] == shown[3] == "[('COMPUTED', 0, 1)]"
