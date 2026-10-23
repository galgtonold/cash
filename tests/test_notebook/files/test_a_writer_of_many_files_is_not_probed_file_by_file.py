"""A statement that wrote more files than a provenance holds is not probed
for every one of them.

``tar.extractall()`` writes every member of the archive. The write
provenance a later reader uses to skip the writer holds at most
``_MAX_PROVENANCE_FILES`` paths, but each written path was probed for
existence first, and only then counted: 100,000 probes per statement, three
statements deep (the ``if``, the ``with``, the call), to learn there were too
many. The count stops once it is past the limit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cash.notebook.statement import records
from cash.notebook.statement.records import StatementRecords
from cash.notebook.tracking_state import TrackingState


class _Backend:
    def __init__(self) -> None:
        self.stored: dict[str, dict] = {}

    def set_metadata_only(self, key, record) -> None:
        self.stored[key] = record


def _records(backend):
    shell = SimpleNamespace(user_ns={})
    return StatementRecords(shell, TrackingState(), SimpleNamespace(backend=backend), None)


@pytest.fixture
def probes(monkeypatch):
    seen = []
    own_stat = records.own_stat

    def counted(path):
        seen.append(path)
        return own_stat(path)

    monkeypatch.setattr(records, "own_stat", counted)
    return seen


def _written(tmp_path, count):
    paths = []
    for i in range(count):
        path = tmp_path / f"{i}.txt"
        path.write_text("x", encoding="utf-8")
        paths.append(str(path))
    return frozenset(paths)


def test_past_the_limit_the_probes_stop_and_nothing_is_recorded(tmp_path, monkeypatch, probes):
    limit = 20
    monkeypatch.setattr(StatementRecords, "_MAX_PROVENANCE_FILES", limit)
    backend = _Backend()
    _records(backend).persist_write_provenance("extract()", set(), None, _written(tmp_path, 500))
    assert len(probes) == limit + 1
    assert backend.stored == {}


def test_up_to_the_limit_every_written_file_is_recorded(tmp_path, monkeypatch, probes):
    monkeypatch.setattr(StatementRecords, "_MAX_PROVENANCE_FILES", 20)
    backend = _Backend()
    written = _written(tmp_path, 20)
    _records(backend).persist_write_provenance("extract()", set(), None, written)
    (record,) = backend.stored.values()
    assert record["paths"] == sorted(written)


def test_a_written_file_gone_again_does_not_count_towards_the_limit(tmp_path, monkeypatch, probes):
    monkeypatch.setattr(StatementRecords, "_MAX_PROVENANCE_FILES", 20)
    backend = _Backend()
    kept = _written(tmp_path, 5)
    gone = frozenset(str(tmp_path / f"tmp{i}") for i in range(100))
    _records(backend).persist_write_provenance("extract()", set(), None, kept | gone)
    (record,) = backend.stored.values()
    assert record["paths"] == sorted(kept)


def test_cashs_own_checks_of_written_paths_are_not_the_statements_probes(tmp_path, monkeypatch):
    """Whether a written path is a folder, and whether it is still there, is
    cash's question: asked past ``os.stat``/``os.lstat`` (which the tracker
    wraps, to record a probe of the statement's), and answered as
    ``os.path`` would."""
    import os

    from cash.notebook import write_observer

    folder = tmp_path / "out"
    folder.mkdir()
    file = tmp_path / "a.txt"
    file.write_text("x", encoding="utf-8")
    dangling = tmp_path / "link"
    try:
        dangling.symlink_to(tmp_path / "gone")
    except OSError:
        dangling = None
    missing = tmp_path / "missing"
    asked = []
    for name in ("stat", "lstat"):
        real = getattr(os, name)

        def wrapper(*a, _real=real, **k):
            asked.append(a)
            return _real(*a, **k)

        wrapper._is_file_tracker_patch = True  # as cash's own wrapper is marked
        wrapper._original_func = real
        monkeypatch.setattr(os, name, wrapper)
    assert write_observer._names_a_folder(str(folder))
    assert not write_observer._names_a_folder(str(file))
    assert not write_observer._names_a_folder(str(missing))
    assert write_observer.written_lstat(str(file)) is not None
    assert write_observer.written_lstat(str(missing)) is None
    if dangling is not None:
        assert write_observer.written_lstat(str(dangling)) is not None, (
            "a dangling link is there, as os.path.lexists says"
        )
    assert not asked, "went through the wrapped os.stat"
