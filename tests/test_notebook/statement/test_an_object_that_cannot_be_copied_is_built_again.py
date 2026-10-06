"""A statement whose output the RAM tier cannot copy is not restored as the
object itself.

The RAM tier keeps a payload it cannot deep-copy (a value holding a lock or a
connection, a chain too deep for deepcopy) by reference. A hit then bound the
very object of the first run, with every change made to it since, where
running the statement builds a fresh one.
"""

from __future__ import annotations

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x"


def _run_all_cells(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


STORE = (
    f"{SETUP}\nimport threading\nclass Store:\n    def __init__(self):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n        self.lock = threading.Lock()\n        self.rows = [1, 2]"
)


def test_an_object_that_cannot_be_copied_is_built_again(cash_magics):
    """The RAM tier keeps a value it cannot copy (here: one holding a lock)
    as the object itself. ``st = Store()`` run again was then a hit binding
    the very object, with the row a later cell appended; plain Python builds
    a fresh one."""
    cells = [STORE, "st = Store()", "st.rows.append(99)"]
    _run_all_cells(cash_magics, cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert cash_magics.shell.user_ns["st"].rows == [1, 2]


def test_a_closed_file_left_by_a_with_block_is_still_restored(cash_magics, statement_processor, tmp_path):
    """Control: the file a ``with open(...)`` leaves behind cannot be copied
    either, but nothing about it can change, so the statement still hits."""
    path = tmp_path / "x.txt"
    path.write_text("hi")
    run_cash_cell(cash_magics, SETUP)
    code = f"with open({str(path)!r}) as f:\n    data = slow(f.read())"
    statement_processor.process_statement(code)
    assert statement_processor.process_statement(code)["status"] == CacheStatus.RESTORED
