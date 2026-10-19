"""A statement that reads and rebinds an input (``n += len(f.read())`` in a
loop over folders) inherits the files that input was built from. They are
taken as the entry the input came from recorded them, as for any input, not
stat'ed and hashed again: by the time the statement stores, the input's
recorded source is the statement's own entry, not yet written, and every
file read before was snapshotted afresh, once per later folder.
"""

from __future__ import annotations

import os

import cash.tracking.file_dep_snapshot as file_dep_snapshot
from cash._paths import normalize_path


def test_files_an_input_was_built_from_are_not_snapshotted_again(
    statement_processor, cash_instance, tmp_path, monkeypatch
):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        folder.mkdir()
        for i in range(3):
            (folder / f"part{i}.txt").write_text(f"{folder.name} {i}", encoding="utf-8")
    snapshotted = []
    real = file_dep_snapshot.snapshot_file_deps
    monkeypatch.setattr(
        file_dep_snapshot,
        "snapshot_file_deps",
        lambda paths, known=None, **kw: snapshotted.append(set(paths)) or real(paths, known, **kw),
    )
    p = statement_processor
    p.process_statement("n = 0")
    read = "n += sum(len(open(os.path.join({!r}, name)).read()) for name in sorted(os.listdir({!r})))"
    p.process_statement("import os")
    p.process_statement(read.format(str(first), str(first)))
    first_entry = cash_instance.backend.peek_metadata(p.tracking_state.variable_sources["n"])
    # As recorded: resolved, with forward slashes on Windows too.
    mark = normalize_path(os.path.realpath(first))
    recorded_first = [paths for paths in snapshotted if any(mark in normalize_path(path) for path in paths)]
    assert recorded_first, snapshotted
    snapshotted.clear()

    meta = p.process_statement(read.format(str(second), str(second)))
    assert meta["status"].name == "COMPUTED"
    again = {path for paths in snapshotted for path in paths if mark in normalize_path(path)}
    assert again == set(), again
    # Still recorded, as the first entry recorded them.
    deps = cash_instance.backend.peek_metadata(p.tracking_state.variable_sources["n"])["file_dependencies"]
    for path, recorded in first_entry["file_dependencies"].items():
        assert deps[path] == recorded
