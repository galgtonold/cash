"""``with cash.assume_safe():`` in a notebook.

The block is for ``@cash.cache`` functions, and it works in one defined in a
cell as in a module. At the top level of a cell a ``with`` statement is one
statement, however many lines it holds, so there the block waives as
``# @cash:assume-safe`` anywhere inside it would: for that statement. The
statement is then cached, which a side effect would otherwise prevent.

Each waiver has its control, the same code without the block.
"""

from __future__ import annotations

import ast
import warnings

import pytest

pytest.importorskip("IPython")

from cash import Cash
from cash.analysis.annotations import get_statement_annotations
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.upstream.replay import StatementReplay
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def magics(cash_magics, mock_shell):
    """``cash_magics`` under ``%cash_on``, with the badge off to keep it out of
    the output and ``cash`` bound in the notebook, as ``import cash`` binds it."""
    import cash

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    mock_shell.user_ns["cash"] = cash
    return cash_magics


def _first(cell: str) -> ast.stmt:
    return ast.parse(cell).body[0]


@pytest.mark.parametrize(
    "cell",
    [
        "with cash.assume_safe():\n    r = save(1)\n",
        "with assume_safe():\n    r = save(1)\n",
        "with cash.assume_safe(), open(p) as fh:\n    r = save(fh.read())\n",
    ],
    ids=["cash", "name", "with-another-item"],
)
def test_the_statement_carries_assume_safe(cell):
    assert get_statement_annotations(cell, _first(cell)).assume_safe
    control = "with open(p) as fh:\n    r = save(fh.read())\n"
    assert not get_statement_annotations(control, _first(control)).assume_safe


def test_an_upstream_repair_reads_it_as_a_direct_run_does():
    """The repair re-runs the statement with the directives its cell gives it."""
    cell = "x = 1\nwith cash.assume_safe():\n    r = save(x)\n"
    found = StatementReplay.statement_directives([cell])
    assert [annotation.assume_safe for annotation in found.values()] == [True]


def _writer_cell(counter, out) -> str:
    """``save`` writes *out* and counts its runs in *counter*; it takes long
    enough to be worth storing."""
    return (
        "import os, time\n"
        "def save(v):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        f"    with open({str(out)!r}, 'w') as fh:\n"
        "        fh.write(str(v))\n"
        f"    fd = os.open({str(counter)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    return v\n"
    )


def test_a_top_level_block_caches_a_statement_with_a_side_effect(magics, tmp_path):
    """``save`` writes a file, so the statement runs every time; inside the
    block, the write is waived and the second run is served from the cache."""
    counter, out = tmp_path / "runs", tmp_path / "out.txt"
    run_cash_cell(magics, _writer_cell(counter, out))
    for _ in range(2):
        run_cash_cell(magics, "r = save(7)\n")
    assert counter.read_bytes() == b"xx", "control: without the block it runs every time"
    for _ in range(2):
        run_cash_cell(magics, "with cash.assume_safe():\n    r = save(8)\n")
    assert counter.read_bytes() == b"xxx"
    assert magics.shell.user_ns["r"] == 8


@pytest.fixture
def cell_runner(mock_shell, tmp_path):
    """The real magic over a disk cache, with that Cash bound to ``c`` so a
    cell can decorate with ``@c.cache``; returns the impurity-warning count."""
    import cash

    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    magics = CashMagics(mock_shell, c)
    magics.cash_on("")
    magics.badges.mode = "off"
    mock_shell.user_ns.update(c=c, cash=cash)

    def run(cell: str) -> int:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_cash_cell(magics, cell)
        return len([w for w in caught if "Impurity" in type(w.message).__name__])

    yield run
    c.backend.clear()


def test_a_function_defined_in_a_cell_honours_the_block(cell_runner):
    cell = (
        "import os\n"
        "@c.cache\n"
        "def audited(n):\n"
        "    with cash.assume_safe():\n"
        "        os.getpid()\n"
        "    return n * 2\n"
        "audited(1)\n"
    )
    assert cell_runner(cell) == 0
    assert cell_runner(cell.replace("audited", "plain").replace("    with cash.assume_safe():\n    ", "")) == 1
