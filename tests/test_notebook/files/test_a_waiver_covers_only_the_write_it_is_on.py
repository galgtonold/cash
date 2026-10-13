"""A waiver covers the write it is on, not every write the statement makes.

``s = export(normalize(x))`` with ``normalize`` marked ``@cash.pure``, and an
export helper that calls a print helper whose line carries
``# @cash:assume-safe``, were stored although ``export`` writes a file: any
waiver or ``@pure`` anywhere in what the statement called was taken for the
write cash saw. The next Run All skipped the write. A waiver on the write's
own line, on a line calling the writer, or on the writer's ``def`` still
caches the statement.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

HELPERS = f"""import gzip, os, time
import cash
COUNTER = None
def count():
    fd = os.open(COUNTER, os.O_WRONLY | os.O_APPEND | os.O_CREAT); os.write(fd, b'x'); os.close(fd)  # @cash:assume-safe
@cash.pure
def normalize(v):
    return v / 2
def export(v, p):
    count(); time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    f = gzip.open(p, 'wt'); f.write(str(v)); f.close()
    return v
def log(msg):
    print(msg)  # @cash:assume-safe
def export_logged(v, p):
    log('exporting')
    return export(v, p)
def export_waived(v, p):
    count(); time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    f = gzip.open(p, 'wt'); f.write(str(v)); f.close()  # @cash:assume-safe
    return v
def export_through(v, p):
    return export(v, p)  # @cash:assume-safe
"""

CASES = {
    "pure argument": ("s = export(normalize(4), P)", 3),
    "waiver on a print helper": ("s = export_logged(4, P)", 3),
    "waiver on the write": ("s = export_waived(4, P)", 1),
    "waiver on the call": ("s = export_through(4, P)", 1),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_write_runs_unless_it_was_waived(cash_magics, tmp_path, case):
    statement, runs = CASES[case]
    counter = tmp_path / "runs"
    out = tmp_path / "out.gz"
    cells = [HELPERS + f"COUNTER = {str(counter)!r}\nP = {str(out)!r}", statement]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert counter.read_bytes() == b"x" * runs
