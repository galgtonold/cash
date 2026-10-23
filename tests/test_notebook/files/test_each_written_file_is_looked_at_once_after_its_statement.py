"""Each file a statement wrote is looked at once after it, not per check.

After a statement that wrote files, three checks asked the disk about each
of them: whether it is still there, for the store's refusal and for the
write provenance, and its stat for the notebook's own-write record. A
statement writing 13,000 files paid for 39,000 asks. They now share what
the first of them found (`StatementExecution.written_lstats`), and what
they decide is as before: a file removed again within the statement is
no write a hit skips, and a link is followed to its file.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

from cash.notebook import write_observer
from cash.notebook.statement import lineage
from cash.tracking.read_classification import stat_key
from tests._cell_driver import run_cash_cell


@pytest.fixture
def asked(monkeypatch):
    """Each written path's asks of the disk after its statement."""
    seen: Counter = Counter()
    real_lstat, real_stat = write_observer.own_lstat, lineage.regular_file_stat

    def own_lstat(path):
        seen[os.fspath(path)] += 1
        return real_lstat(path)

    def regular_file_stat(path):
        seen[os.fspath(path)] += 1
        return real_stat(path)

    monkeypatch.setattr(write_observer, "own_lstat", own_lstat)
    monkeypatch.setattr(lineage, "regular_file_stat", regular_file_stat)
    return seen


def _run(cash_magics, cells):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


def test_a_written_file_is_asked_about_once(cash_magics, tmp_path, asked):
    out = tmp_path / "out"
    out.mkdir()
    names = [str(out / f"f{i}.txt") for i in range(20)]
    cells = [f"import pathlib\nn = sum(pathlib.Path({str(out)!r}, f'f{{i}}.txt').write_text('x') for i in range(20))"]
    _run(cash_magics, cells)
    assert cash_magics.shell.user_ns["n"] == 20
    assert {name: asked[name] for name in names} == dict.fromkeys(names, 1)
    record = cash_magics.tracking_state.own_file_writes
    real = {os.path.realpath(name) for name in names}
    assert real <= set(record)
    for name in names:
        assert record[os.path.realpath(name)][1] == stat_key(os.stat(name))


def test_a_file_removed_within_the_statement_is_no_kept_write(cash_magics, tmp_path):
    scratch = tmp_path / "scratch.txt"
    cells = [f"import os\np = {str(scratch)!r}\nn = [open(p, 'w').write('x'), os.remove(p)]"]
    _run(cash_magics, cells)
    assert os.path.realpath(scratch) not in cash_magics.tracking_state.own_file_writes


def test_a_link_written_through_records_its_file(cash_magics, tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("no symbolic links here")
    cells = [f"n = open({str(link)!r}, 'w').write('abc')"]
    _run(cash_magics, cells)
    assert cash_magics.tracking_state.own_file_writes[os.path.realpath(link)][1] == stat_key(os.stat(target))
