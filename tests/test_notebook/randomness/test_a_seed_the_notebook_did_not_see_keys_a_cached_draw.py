"""A seed the notebook's ledger did not see still keys a ``@cash.cache`` draw.

The statement engine publishes its seed ledger, and a ``@cash.cache``
function that draws from the global stream keyed on it. The ledger only
learns of seeds from the statements the engine runs: after ``%cash_off``,
``np.random.seed(2)`` runs as plain Python and the ledger still holds the
seed-1 statement's epoch, so ``f(1)`` was served the draw made under seed
1. The same happened in any process that once ran a notebook in-process
(the edit benchmark) and then called cached functions: the dead engine's
ledger stayed published, and a full test run failed three decorator tests
that ran after ``test_edit_effectiveness_floor.py`` on the same worker.

The engine now publishes, with the ledger, which seeds it accounts for; a
seed set since keys the stream by where it stands, as outside a notebook.
"""

from __future__ import annotations

import numpy as np

from cash import Cash
from cash.tracking.randomness import lineage
from tests._cell_driver import run_cash_cell


def drawn(x):
    return x + float(np.random.normal())


def _served_and_plain(f, seeds):
    """``(f(1), drawn(1))`` per seed, each call made right after seeding."""
    out = []
    for s in seeds:
        np.random.seed(s)
        plain = drawn(1)
        np.random.seed(s)
        out.append((f(1), plain))
    return out


def test_a_seed_set_after_cash_off_keys_the_draw(cash_magics):
    run_cash_cell(cash_magics, "import numpy as np\nnp.random.seed(1)")
    assert "numpy.random" in lineage.seed_epochs(), "the ledger saw no seed: the test proves nothing"
    cash_magics.cash_off("")
    f = Cash(backend="memory", register_magic=False).cache(drawn)
    # The first calls learn that f draws, and store its seed-1 value.
    for served, plain in _served_and_plain(f, (1, 1, 1, 2, 1, 2)):
        assert served == plain


def test_a_seed_set_after_the_notebook_is_gone_keys_the_draw(cash_magics):
    """The engine of a notebook run in this process outlives it: nothing
    unpublishes its ledger."""
    run_cash_cell(cash_magics, "import numpy as np\nnp.random.seed(1)")
    del cash_magics
    f = Cash(backend="memory", register_magic=False).cache(drawn)
    for served, plain in _served_and_plain(f, (1, 1, 1, 2, 1, 2)):
        assert served == plain
