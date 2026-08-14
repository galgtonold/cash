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
    """hint() must guarantee TRUE ASCII for badge embedding, not merely
    cp1252-safety: the downstream reader's codepage is unknown, and cp437 /
    cp850 consoles exist too. A cp1252-representable accented character
    passing through unescaped would still crash one of those. Replace
    non-encodable chars rather than lose the hint entirely."""
    nb = _touch(tmp_path / "nb.ipynb")
    t = StalenessTracker()
    # A cell with non-ASCII characters (café, café spelled with different encoding)
    t.observe(running_code="café = 1\nmore()", file_code="cafe = 1",
              notebook_path=str(nb))
    hint = t.hint()
    assert hint is not None
    # The real constraint: must be encodable as pure ASCII.
    hint.encode("ascii")  # should not raise
    # A cp1252 encode is a corollary of the ASCII one (ASCII is a
    # subset of cp1252), kept as a belt-and-suspenders check.
    hint.encode("cp1252")  # should not raise
    # ASCII parts should survive
    assert "=" in hint
    assert "1" in hint


def test_percent_percent_cash_cell_untouched_is_not_stale(tmp_path):
    """CAS-274 final-review finding 1 (CRITICAL). `%%cash` is a cell magic:
    IPython strips the `%%cash` line before handing the cell's BODY to cash
    (see `magics.py`'s `cash()`), but the file on disk still has that line --
    it is what the user actually saved. Comparing the bare body against
    "magic line + body" made every untouched, perfectly-saved `%%cash` cell
    look edited. This is the false positive the fix must eliminate."""
    nb = _touch(tmp_path / "nb.ipynb", "%%cash\nTHRESHOLD = 0.5")
    t = StalenessTracker()
    assert t.observe(
        running_code="THRESHOLD = 0.5",             # the BODY IPython hands cash
        file_code="%%cash\nTHRESHOLD = 0.5",         # the FULL text the file has
        notebook_path=str(nb),
    ) is False
    assert t.is_stale() is False


def test_percent_percent_cash_cell_with_ttl_arg_untouched_is_not_stale(tmp_path):
    """The magic line can carry args (`%%cash ttl=60`); the fix must not
    depend on the bare `%%cash` spelling with nothing after it."""
    nb = _touch(tmp_path / "nb.ipynb", "%%cash ttl=60\nTHRESHOLD = 0.5")
    t = StalenessTracker()
    assert t.observe(
        running_code="THRESHOLD = 0.5",
        file_code="%%cash ttl=60\nTHRESHOLD = 0.5",
        notebook_path=str(nb),
    ) is False
    assert t.is_stale() is False


def test_percent_percent_cash_cell_real_edit_is_still_caught(tmp_path):
    """Positive control for the two tests above: stripping the `%%cash` line
    from the comparison must not blind the detector to a genuine edit of the
    cell's body -- the whole point is to keep detection for %%cash cells,
    not just silence it."""
    nb = _touch(tmp_path / "nb.ipynb", "%%cash\nTHRESHOLD = 0.5")
    t = StalenessTracker()
    assert t.observe(
        running_code="THRESHOLD = 0.9",              # edited body, unsaved
        file_code="%%cash\nTHRESHOLD = 0.5",          # file still has the old value
        notebook_path=str(nb),
    ) is True
    assert t.is_stale() is True


def test_percent_cash_on_hook_path_unaffected_by_the_magic_line_strip(tmp_path):
    """Guard test for the fix's safety condition: the `%cash_on` hook path
    hands cash the RAW, unprocessed cell, so when running_code ALREADY
    starts with a `%%` line, nothing should be stripped from file_code --
    stripping only one side would turn a real edit invisible or, worse,
    manufacture a mismatch out of nothing."""
    nb = _touch(tmp_path / "nb.ipynb", "%%time\nTHRESHOLD = 0.5")
    t = StalenessTracker()
    # Identical on both sides -- must not be reported stale.
    assert t.observe(
        running_code="%%time\nTHRESHOLD = 0.5",
        file_code="%%time\nTHRESHOLD = 0.5",
        notebook_path=str(nb),
    ) is False
    # A real edit on both sides -- must still be caught.
    t2 = StalenessTracker()
    assert t2.observe(
        running_code="%%time\nTHRESHOLD = 0.9",
        file_code="%%time\nTHRESHOLD = 0.5",
        notebook_path=str(nb),
    ) is True


def test_checker_no_false_positive_on_saved_percent_percent_cash_cell(tmp_path):
    """Checker-level regression test, in the shape the review asked for:
    drives `_find_current_cell_index` against a REAL `.ipynb` on disk (no
    kernel needed), reading cell sources the way
    `get_notebook_cells_with_ids` actually returns them --
    `"".join(source_lines)`, magic line and all -- rather than a hand-typed
    string that might not match nbformat's own join behaviour."""
    import json

    from cash.notebook.server_discovery import get_notebook_cells_with_ids
    from cash.notebook.upstream.checker import UpstreamChecker

    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "id": "c1",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["%%cash\n", "THRESHOLD = 0.5"],
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }), encoding="utf-8")

    cells_with_ids = get_notebook_cells_with_ids(str(nb))
    assert cells_with_ids == [("c1", "%%cash\nTHRESHOLD = 0.5")]  # sanity on the fixture

    checker = UpstreamChecker.__new__(UpstreamChecker)   # no full init needed
    checker.debug = False
    checker.staleness = StalenessTracker()
    checker._notebook_path_for_staleness = str(nb)

    # What the %%cash cell magic actually hands execute_cell: the BODY only
    # (magics.py's `cash()` cell-magic method, per IPython's own contract).
    idx = checker._find_current_cell_index(
        "THRESHOLD = 0.5",
        [src for _, src in cells_with_ids],
        cell_id="c1",
        cells_with_ids=cells_with_ids,
    )
    assert idx == 0
    assert checker.staleness.is_stale() is False, (
        "an untouched, saved %%cash cell must never be reported stale"
    )


def test_checker_still_catches_a_real_edit_in_a_percent_percent_cash_cell(tmp_path):
    """Positive control for the checker-level test above, same real-.ipynb
    shape: an unsaved edit to a %%cash cell's body must still be caught."""
    import json

    from cash.notebook.server_discovery import get_notebook_cells_with_ids
    from cash.notebook.upstream.checker import UpstreamChecker

    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "id": "c1",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["%%cash\n", "THRESHOLD = 0.5"],
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }), encoding="utf-8")

    cells_with_ids = get_notebook_cells_with_ids(str(nb))

    checker = UpstreamChecker.__new__(UpstreamChecker)
    checker.debug = False
    checker.staleness = StalenessTracker()
    checker._notebook_path_for_staleness = str(nb)

    # The editor has THRESHOLD = 0.9 (unsaved); the file on disk still says 0.5.
    idx = checker._find_current_cell_index(
        "THRESHOLD = 0.9",
        [src for _, src in cells_with_ids],
        cell_id="c1",
        cells_with_ids=cells_with_ids,
    )
    assert idx == 0
    assert checker.staleness.is_stale() is True


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
