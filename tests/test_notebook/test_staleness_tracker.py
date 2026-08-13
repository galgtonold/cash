"""The saved .ipynb can be PROVEN out of date, and cash already has the proof.

When IPython hands cash the code it is running and the file claims that same
cell says something else, the file is stale -- and so is every other cell cash
read from it. This tracker holds that verdict for the session.
"""
from __future__ import annotations

import os
import time

from cash.notebook.staleness import StalenessTracker


def _touch(path, text="x"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_matching_source_is_not_stale(tmp_path):
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    assert t.observe(running_code="a = 1", file_code="a = 1", notebook_path=str(nb)) is False
    assert t.is_stale() is False


def test_differing_source_proves_the_file_is_stale(tmp_path):
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    assert t.observe(running_code="a = 2", file_code="a = 1", notebook_path=str(nb)) is True
    assert t.is_stale() is True
    assert t.saved_at() == os.stat(nb).st_mtime


def test_whitespace_only_difference_is_not_stale(tmp_path):
    """Editors and nbformat disagree about trailing newlines constantly. A
    trailing-newline difference is not evidence of an unsaved edit, and firing
    on it would make the warning noise the user learns to ignore."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    assert t.observe(running_code="a = 1\n", file_code="a = 1", notebook_path=str(nb)) is False
    assert t.is_stale() is False


def test_the_verdict_persists_after_a_later_matching_cell(tmp_path):
    """One proof condemns the whole FILE, not one cell. A later cell that
    happens to match does not clear it -- cash still read the others from a
    file it now knows was out of date."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    t.observe(running_code="a = 2", file_code="a = 1", notebook_path=str(nb))
    t.observe(running_code="b = 1", file_code="b = 1", notebook_path=str(nb))
    assert t.is_stale() is True


def test_saving_the_file_clears_the_verdict(tmp_path):
    """The mtime moving is the only evidence that the file caught up."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    t.observe(running_code="a = 2", file_code="a = 1", notebook_path=str(nb))
    assert t.is_stale() is True

    time.sleep(0.01)
    _touch(nb, "y")                       # a save moves mtime
    t.observe(running_code="b = 1", file_code="b = 1", notebook_path=str(nb))
    assert t.is_stale() is False


def test_a_missing_file_is_never_stale(tmp_path):
    """Degrade, never raise: no file means no proof, not a problem."""
    t = StalenessTracker()
    assert t.observe(running_code="a = 2", file_code="a = 1",
                     notebook_path=str(tmp_path / "gone.ipynb")) is False
    assert t.is_stale() is False


def test_absent_file_code_is_never_stale(tmp_path):
    """cash could not locate this cell in the file. Absence of evidence is not
    evidence -- notably, a cell added since the last save has no file copy and
    must not be reported as proof of staleness."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    assert t.observe(running_code="a = 2", file_code=None, notebook_path=str(nb)) is False
    assert t.is_stale() is False


def test_hint_names_the_running_cell(tmp_path):
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    t.observe(running_code="THRESHOLD = 0.9\nmore()", file_code="THRESHOLD = 0.5",
              notebook_path=str(nb))
    assert "THRESHOLD = 0.9" in t.hint()


def test_re_notify_guard_does_not_re_emit_on_still_mismatching_cell(tmp_path):
    """A second mismatching cell should not re-emit; the transition happened once."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    assert t.observe(running_code="a = 2", file_code="a = 1", notebook_path=str(nb)) is True
    saved_at_first = t.saved_at()
    hint_first = t.hint()

    # Second call with STILL-MISMATCHING code should not re-notify
    assert t.observe(running_code="b = 2", file_code="b = 1", notebook_path=str(nb)) is False
    # and verdict details must not change
    assert t.saved_at() == saved_at_first
    assert t.hint() == hint_first


def test_hint_handles_non_ascii_by_replacement(tmp_path):
    """hint() must guarantee ASCII for badge embedding. Replace non-encodable chars."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    # A cell with non-ASCII characters (café, café spelled with different encoding)
    t.observe(running_code="café = 1\nmore()", file_code="cafe = 1",
              notebook_path=str(nb))
    hint = t.hint()
    assert hint is not None
    # Must be encodable as cp1252 (the downstream constraint)
    hint.encode("cp1252")  # should not raise
    # ASCII parts should survive
    assert "=" in hint
    assert "1" in hint


def test_checker_detects_a_stale_file_on_an_id_match(tmp_path, monkeypatch):
    """The proof site. `_find_current_cell_index` matches by cell ID and returns
    immediately -- both strings are in hand at that moment and were discarded.
    """
    from cash.notebook.upstream.checker import UpstreamChecker

    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")

    checker = UpstreamChecker.__new__(UpstreamChecker)   # no full init needed
    checker.debug = False
    checker.staleness = StalenessTracker()
    checker._notebook_path_for_staleness = str(nb)

    idx = checker._find_current_cell_index(
        "THRESHOLD = 0.9",
        ["THRESHOLD = 0.5"],
        cell_id="c1",
        cells_with_ids=[("c1", "THRESHOLD = 0.5")],
    )
    assert idx == 0, "the ID match must still win; detection is a side effect"
    assert checker.staleness.is_stale() is True


def test_checker_is_quiet_when_the_file_matches(tmp_path):
    """Control. Without this, a detector that fired unconditionally passes."""
    from cash.notebook.upstream.checker import UpstreamChecker

    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")

    checker = UpstreamChecker.__new__(UpstreamChecker)
    checker.debug = False
    checker.staleness = StalenessTracker()
    checker._notebook_path_for_staleness = str(nb)

    checker._find_current_cell_index(
        "THRESHOLD = 0.5",
        ["THRESHOLD = 0.5"],
        cell_id="c1",
        cells_with_ids=[("c1", "THRESHOLD = 0.5")],
    )
    assert checker.staleness.is_stale() is False


def test_check_and_reexecute_stores_the_resolved_notebook_path():
    """Regression guard for a NO-OP, which the tests above cannot see.

    `observe()` returns False whenever `notebook_path` is None, so a checker
    that never stores the resolved path detects nothing in production while
    every unit test still passes — the feature would ship dead.

    Asserting on the source of the one method that must do the wiring, rather
    than on the file's text, so renaming a local does not break the test while
    deleting the wiring still does.
    """
    import inspect

    from cash.notebook.upstream.checker import UpstreamChecker

    src = inspect.getsource(UpstreamChecker.check_and_reexecute)
    assert "_notebook_path_for_staleness" in src, (
        "check_and_reexecute never stores the resolved notebook path, so "
        "StalenessTracker.observe always receives notebook_path=None and can "
        "never report staleness"
    )
