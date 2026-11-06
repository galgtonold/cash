"""A file a statement wrote is stat-ed once as it ends, not twice.

The check that a written file is still there (`kept_writes`, which keeps the
statement from being stored) took an ``lstat`` of each, and the record of
the notebook's own writes (`note_own_writes`) a ``stat`` of each again: a
tar extraction of 50,000 files paid both, ~2 s on Windows. The record now
takes the check's ``lstat`` of a regular file that is no link.

What must hold: the record is the file's stat after the write; a statement
whose writes were not checked (it is not stored for another reason) still
stats each file itself.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook.statement import lineage
from tests._cell_driver import run_cash_cell

N = 30

#: What the static writer check does not know: the statement is checked for
#: what it left, as ``tar.extractall()`` is.
EXTRACT = """with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
    tar.extractall("out", **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))"""


#: The archive, made by a cell: an input with a lineage, as a file read would give.
MAKE = f"""import io, tarfile
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as made:
    for i in range({N}):
        info = tarfile.TarInfo(f"{{i}}.txt")
        info.size = i
        made.addfile(info, io.BytesIO(b"x" * i))
blob = buf.getvalue()"""


@pytest.fixture
def stats_by_the_record(cash_magics, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    monkeypatch.chdir(tmp_path)
    taken = []
    real = lineage.regular_file_stat

    def counting(path):
        taken.append(path)
        return real(path)

    monkeypatch.setattr(lineage, "regular_file_stat", counting)
    run_cash_cell(cash_magics, MAKE)
    taken.clear()
    return taken


def _recorded(cash_magics, tmp_path) -> dict[str, tuple]:
    own = cash_magics.tracking_state.own_file_writes
    return {os.path.basename(p): rec[1] for p, rec in own.items() if os.path.dirname(p).endswith("out")}


def _stat_key(path) -> tuple:
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns, getattr(st, "st_ctime_ns", 0))


def test_the_record_takes_the_check_s_lstat(cash_magics, tmp_path, stats_by_the_record):
    run_cash_cell(cash_magics, EXTRACT)
    recorded = _recorded(cash_magics, tmp_path)
    assert len(recorded) == N
    assert all(recorded[f"{i}.txt"] == _stat_key(tmp_path / "out" / f"{i}.txt") for i in range(N))
    assert stats_by_the_record == []


def test_a_statement_whose_writes_were_not_checked_stats_them(cash_magics, tmp_path, stats_by_the_record):
    """Reading the clock keeps it from being stored before its writes are checked."""
    run_cash_cell(cash_magics, "import time\n" + EXTRACT + "\n    t = time.time()")
    recorded = _recorded(cash_magics, tmp_path)
    assert len(recorded) == N
    assert all(recorded[f"{i}.txt"] == _stat_key(tmp_path / "out" / f"{i}.txt") for i in range(N))
    assert len(stats_by_the_record) >= N, "control: the record stats what was not checked"
