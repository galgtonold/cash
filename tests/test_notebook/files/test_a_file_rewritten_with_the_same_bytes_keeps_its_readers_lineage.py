"""A file rewritten with the same bytes keeps the lineage of what reads it,
whether the reader is restored or runs again.

The file part of a lineage is the file's stat. A data export run again
rewrote ``meters.csv`` unchanged: the reader's stored entry passed its
content check and was restored with the lineage it was stored with, but
after a restart the reader -- too quick to keep on disk -- ran again under
the new stat, and every statement built on it keyed differently (a t-SNE
run again, 8 s, with its input identical). The content check now records
the new stat as a version of the old one, and the lineage names the file
by the old one.
"""

from __future__ import annotations

import os

from cash.notebook.statement.file_deps import compute_file_hash_component
from cash.tracking import file_versions
from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_file_deps
from tests._cell_driver import run_cash_cell


def _written_before_this_process(monkeypatch):
    """The files count as written before the process started: no hashing on
    their stat alone (`file_versions.content_version`), only the content check
    links a rewrite."""
    monkeypatch.setattr(file_versions, "_STARTED_NS", 1 << 62)


def _rewrite(path, text):
    """Write *text* again, with a timestamp that differs from the last one."""
    before = os.stat(path).st_mtime_ns
    path.write_text(text, encoding="utf-8", newline="\n")
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def test_a_content_check_names_the_new_stat_by_the_old(tmp_path, monkeypatch):
    _written_before_this_process(monkeypatch)
    data = tmp_path / "meters.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8", newline="\n")
    snapshot = snapshot_file_deps({str(data)})[str(data)]
    before = compute_file_hash_component({str(data)})

    _rewrite(data, "a,b\n1,2\n")
    assert compute_file_hash_component({str(data)}) != before, "control: the stat moved"
    assert file_dep_is_fresh(str(data), snapshot) == (True, None)

    assert compute_file_hash_component({str(data)}) == before


def test_new_bytes_still_move_the_lineage(tmp_path, monkeypatch):
    _written_before_this_process(monkeypatch)
    data = tmp_path / "meters.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8", newline="\n")
    snapshot = snapshot_file_deps({str(data)})[str(data)]
    before = compute_file_hash_component({str(data)})
    _rewrite(data, "a,b\n1,2\n")
    file_dep_is_fresh(str(data), snapshot)

    _rewrite(data, "a,b\n9,9\n")

    assert file_dep_is_fresh(str(data), snapshot)[0] is False
    assert compute_file_hash_component({str(data)}) != before


def test_a_reader_run_again_takes_the_lineage_it_was_restored_with(cash_magics, clean_backend, tmp_path):
    data = tmp_path / "meters.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8", newline="\n")
    cash_magics.cash_persist("on")
    read = f"rows = open({str(data)!r}).read().splitlines()"
    lineage = cash_magics.tracking_state.variable_lineage

    run_cash_cell(cash_magics, read)
    first = lineage["rows"]
    _rewrite(data, "a,b\n1,2\n")
    run_cash_cell(cash_magics, read)
    assert lineage["rows"] == first, "control: restored with the lineage it was stored with"

    clean_backend.clear()
    cash_magics.tracking_state.reset_session_state()
    cash_magics.shell.user_ns.pop("rows", None)
    run_cash_cell(cash_magics, read)

    assert cash_magics.shell.user_ns["rows"] == ["a,b", "1,2"]
    assert lineage["rows"] == first
