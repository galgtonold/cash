"""A file is checked once per cell, not once per statement, until code runs.

A cell whose statements all derive from 10,000 documents
re-checked every document for every statement lookup -- 120,000 checks, 1.1 s,
in a cell served entirely from the cache. The answer for a file holds until a
statement executes (it may write the file) or the next cell begins.
"""

import types

from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_file_deps


class _Backend:
    def __init__(self, deps):
        self._meta = {"key": "stmt:x", "file_dependencies": deps}

    def get(self, key):
        return dict(self._meta, key=key), "value"


def _setup(tmp_path, monkeypatch, n=20):
    paths = []
    for i in range(n):
        p = tmp_path / f"d{i:02d}.txt"
        p.write_text(f"doc {i}\n")
        paths.append(str(p))
    checks = []
    real = file_dep_snapshot.file_dep_is_fresh

    def counting(*args, **kwargs):
        checks.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(file_dep_snapshot, "file_dep_is_fresh", counting)
    checker = CacheFreshnessChecker(_Backend(snapshot_file_deps(set(paths))))
    state = types.SimpleNamespace(executed_file_deps={}, variable_sources={})
    return paths, checks, checker, state


def test_statements_of_one_cell_share_the_answers(tmp_path, monkeypatch):
    paths, checks, checker, state = _setup(tmp_path, monkeypatch)
    for key in ("stmt:a", "stmt:b", "stmt:c"):
        _, data, _ = checker.check_cache(state, key, None, epoch=7)
        assert data == "value"
    assert len(checks) == len(paths)


def test_the_next_cell_checks_again(tmp_path, monkeypatch):
    paths, checks, checker, state = _setup(tmp_path, monkeypatch)
    checker.check_cache(state, "stmt:a", None, epoch=7)
    with open(paths[3], "w") as fh:
        fh.write("edited between cells\n")
    _, data, _ = checker.check_cache(state, "stmt:a", None, epoch=8)
    assert data is None
    assert "d03.txt" in (checker.last_miss_reason or "")


def test_a_statement_that_runs_makes_the_next_lookup_check_again(tmp_path, monkeypatch):
    paths, checks, checker, state = _setup(tmp_path, monkeypatch)
    checker.check_cache(state, "stmt:a", None, epoch=7)
    checker.forget_file_answers(7)  # what executing a statement does
    with open(paths[5], "w") as fh:
        fh.write("written by the statement that ran\n")
    _, data, _ = checker.check_cache(state, "stmt:b", None, epoch=7)
    assert data is None


def test_outside_a_cell_every_lookup_checks(tmp_path, monkeypatch):
    """No execution count to tie the answers to (a processor driven directly):
    a file changed between two lookups is seen, as before."""
    paths, checks, checker, state = _setup(tmp_path, monkeypatch)
    checker.check_cache(state, "stmt:a", None, epoch=None)
    with open(paths[0], "w") as fh:
        fh.write("changed between lookups\n")
    _, data, _ = checker.check_cache(state, "stmt:a", None, epoch=None)
    assert data is None


def test_a_set_already_found_fresh_is_not_walked_again(tmp_path, monkeypatch):
    """A 25-iteration loop over a frame read from 5,226
    files. Each file was checked once, but every lookup still walked all 5,226
    answers -- a million calls, 2.4 s of a cell that takes 1.1 s plain. A
    dependency set found fresh whole is not walked again until the answers
    lapse."""
    paths, checks, checker, state = _setup(tmp_path, monkeypatch, n=100)
    walked = []
    real = file_dep_snapshot.dep_is_fresh
    monkeypatch.setattr(file_dep_snapshot, "dep_is_fresh", lambda *a, **k: walked.append(a[0]) or real(*a, **k))
    for key in ("stmt:a", "stmt:b", "stmt:c"):
        _, data, _ = checker.check_cache(state, key, None, epoch=7)
        assert data == "value"
    assert len(walked) == len(paths), f"walked {len(walked)} answers for {len(paths)} files"

    checker.forget_file_answers(7)  # a statement ran: check again
    with open(paths[42], "w") as fh:
        fh.write("written by the statement that ran\n")
    _, data, _ = checker.check_cache(state, "stmt:d", None, epoch=7)
    assert data is None
