"""`# @cash:assume-safe` must work for a function defined in a cell.

cash splits a cell with ``ast.unparse``, which strips comments, then compiles
that text and registers it in linecache. So ``inspect.getsource`` on a function
defined in a cell returned source with no comments in it, and the per-line
purity waiver -- read back from the function's own source -- was invisible.

Measured before the fix:

    in a module file      annotated 0 warnings, unannotated 1
    in a %cash_on cell    annotated 1 warning,  unannotated 1

Both arms are required. Without the unannotated arm, a run where warnings are
broken generally would look like a pass.
"""
from __future__ import annotations

import tempfile
import warnings

import pytest

pytest.importorskip("IPython")

from cash import Cash
from cash.notebook.ipython.magics import CashMagics
from tests.conftest import MockShell


@pytest.fixture
def cell_runner():
    """Run cells through the real magic, returning the impurity-warning count."""
    shell = MockShell()
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    magics = CashMagics(shell, cash)
    magics.cash_on("")
    magics._badge_mode = "off"          # keep the badge out of the captured output
    shell.user_ns["c"] = cash

    def run(cell: str) -> int:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            magics.cash("", cell)
        return len([w for w in caught if "Impurity" in type(w.message).__name__])

    return run


ANNOTATED = (
    "import time\n"
    "@c.cache\n"
    "def audited(n):\n"
    "    time.sleep(0.01)  # @cash:assume-safe - deliberate, this is the test\n"
    "    return n * 2\n"
    "audited(1)\n"
)

PLAIN = (
    "import time\n"
    "@c.cache\n"
    "def unaudited(n):\n"
    "    time.sleep(0.01)\n"
    "    return n * 2\n"
    "unaudited(1)\n"
)


def test_an_annotated_line_is_waived_in_a_notebook_cell(cell_runner):
    assert cell_runner(ANNOTATED) == 0, (
        "the waiver was not honoured for a function defined in a cell"
    )


def test_an_unannotated_line_still_warns_in_a_notebook_cell(cell_runner):
    """The control: proves the test above is not passing because warnings are off."""
    assert cell_runner(PLAIN) == 1
