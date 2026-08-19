"""A proven-stale file must be VISIBLE, not silently worked around.

The badge already has a home for this: notification rows, the same family as
FUNC CHANGED and MODULE RELOADED, which render on the red rail and add to the
`warn` chip. It deliberately does NOT go in the header status word -- per
CAS-273 that binary is near-permanently EXECUTED, so a fourth meaning would
bury the one thing here worth reading.
"""
from __future__ import annotations

import types

from cash.notebook.ipython.cell_executor import CellExecutor, staleness_notification
from cash.notebook.staleness import StalenessTracker


def _stale_tracker(tmp_path):
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")
    t = StalenessTracker()
    t.observe(running_code="THRESHOLD = 0.9", file_code="THRESHOLD = 0.5",
              notebook_path=str(nb))
    return t


def test_a_fresh_file_produces_no_notification(tmp_path):
    """The control, and it is the important one: this row must appear only when
    cash actually has proof, or it becomes noise the user tunes out."""
    assert staleness_notification(StalenessTracker()) is None


def test_a_stale_file_produces_a_warning_row(tmp_path):
    n = staleness_notification(_stale_tracker(tmp_path))
    assert n is not None
    assert n["status"] == "WARNING"
    assert n["is_upstream"] is True


def test_the_message_names_the_evidence_and_the_remedy(tmp_path):
    """Naming the save time is what makes it checkable rather than nagging, and
    naming the remedy is what makes it actionable."""
    n = staleness_notification(_stale_tracker(tmp_path))
    assert "saved at" in n["code"]
    assert "Save" in n["code"]
    assert "THRESHOLD = 0.9" in n["code"]


def test_the_message_survives_a_legacy_console(tmp_path):
    """`%cash_badge print` is read by a DIFFERENT process than wrote it --
    nbconvert, a log scraper, an agent -- on a possibly cp1252 console. An
    emoji here crashes that reader instead of showing it the badge."""
    n = staleness_notification(_stale_tracker(tmp_path))
    n["code"].encode("cp1252")


def test_the_message_shows_the_running_code_not_the_file_code(tmp_path):
    """Pins the observe() argument mapping (carried forward from the Task 2
    review). StalenessTracker's comparison is symmetric, so swapping
    running_code and file_code at the call site would still make is_stale()
    True and still pass every test above -- but hint() would then hold the
    FILE's stale text while this message presents it as the code that is
    running. This notification renders hint() straight to the user, so an
    inverted mapping would show them exactly the wrong line."""
    n = staleness_notification(_stale_tracker(tmp_path))
    assert "THRESHOLD = 0.9" in n["code"]        # the RUNNING code
    assert "THRESHOLD = 0.5" not in n["code"]    # NOT the file's stale copy


def test_the_remedy_survives_print_modes_80_char_cap(tmp_path):
    """CAS-274 final-review finding 2 (IMPORTANT). `%cash_badge print` renders
    `code` through `renderers.text._row_line`, which hard-truncates a row's
    first line at `theme.HEADER_MAX_LEN` (80 chars) -- unlike HTML, there is
    no tooltip or drawer in that mode to hold the rest. The old message put
    the actionable half ("Save and re-run") at the END, past the cap, so
    print mode -- exactly the nbconvert / log-scraper / agent audience this
    feature was built for -- showed a warning with no remedy.

    Renders through the REAL text renderer (not the raw dict) so a fix that
    only reorders dict keys without actually fitting the 80-char cap still
    fails this, and asserts on "ctrl"/"re-run" rather than "save" alone,
    since "save" is a substring of "saved" and would spuriously match the
    unfixed message's "...notebook saved at..." clause.
    """
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge

    n = staleness_notification(_stale_tracker(tmp_path))
    view = build_interactive_badge([n])
    text = render_text(view)
    lowered = text.lower()

    assert "stale" in lowered, f"the fact of staleness did not survive truncation: {text!r}"
    assert "ctrl" in lowered, f"the remedy (Ctrl+S) did not survive truncation: {text!r}"
    assert "re-run" in lowered, f"the remedy (re-run) did not survive truncation: {text!r}"


def test_the_executor_actually_calls_it_with_a_live_tracker(tmp_path):
    """Wiring regression guard.

    `staleness_notification()` passing in isolation proves nothing if the
    executor never calls it against a real tracker -- which is exactly what
    happened in an earlier draft of this wiring: the obvious call site
    (`self._statement_processor.upstream_checker.staleness`) does not exist
    on `CellExecutor` and raises AttributeError on every single cell run.
    The correct handle is `self._upstream_checker` (set in __init__). This
    also pins WHERE the check runs: `_build_pre_execution_notifications` is
    called once, after upstream resolution has already run for the current
    cell (unlike `_detect_module_changes`, which runs before it and would see
    only the previous cell's verdict -- see the comment at the call site).
    """
    executor = CellExecutor.__new__(CellExecutor)
    executor._upstream_checker = types.SimpleNamespace(staleness=_stale_tracker(tmp_path))
    executor._statement_processor = types.SimpleNamespace(function_tracker=object())
    executor.shell = types.SimpleNamespace(user_ns={})
    executor._debug = False

    all_metrics = executor._build_pre_execution_notifications("x = 1", [], [])

    warnings = [m for m in all_metrics if m.get("status") == "WARNING"]
    assert len(warnings) == 1
    assert "THRESHOLD = 0.9" in warnings[0]["code"]


def test_a_raising_tracker_does_not_crash_the_notification_builder(tmp_path):
    """The staleness check is a diagnostic nicety layered onto upstream
    resolution, which must already have succeeded to reach this point.
    Nothing in StalenessTracker raises today, but a future break in it must
    degrade to "no warning" -- never take the user's cell execution down over
    what is, at worst, a missed notification. Unlike its guarded siblings
    (`_make_function_change_metrics`, `_make_opaque_warning_metrics`), which
    catch a short, specific exception tuple, this one has no particular
    failure mode to anticipate, so it must be unconditional: this test raises
    something outside either sibling's tuple to prove the guard is not
    accidentally narrow."""
    class _ExplodingTracker:
        def is_stale(self):
            raise RuntimeError("boom")

    executor = CellExecutor.__new__(CellExecutor)
    executor._upstream_checker = types.SimpleNamespace(staleness=_ExplodingTracker())
    executor._statement_processor = types.SimpleNamespace(function_tracker=object())
    executor.shell = types.SimpleNamespace(user_ns={})
    executor._debug = False

    all_metrics = executor._build_pre_execution_notifications("x = 1", [], [])  # must not raise

    assert not any(m.get("status") == "WARNING" for m in all_metrics)


def test_an_unverifiable_source_is_announced_once(tmp_path):
    """Loud, but once. A row on every cell forever is noise users tune out --
    and this warning has to be believed the once it matters."""
    from cash.notebook.ipython.cell_executor import unverifiable_notification
    from cash.notebook.staleness import StalenessTracker

    t = StalenessTracker()
    t.note_source("file")
    first = unverifiable_notification(t)
    assert first is not None
    assert first["status"] == "WARNING"
    assert "cannot" in first["code"].lower()

    assert unverifiable_notification(t) is None, "announced twice in one session"


def test_a_verifiable_source_says_nothing(tmp_path):
    """The control. Colab and the VS Code backup both give live cells, so there
    is nothing to warn about and the row must not appear."""
    from cash.notebook.ipython.cell_executor import unverifiable_notification
    from cash.notebook.staleness import StalenessTracker

    for source in ("colab", "vscode-backup"):
        t = StalenessTracker()
        t.note_source(source)
        assert unverifiable_notification(t) is None, source
        assert t.can_verify() is True, source


def test_the_unverifiable_message_survives_a_legacy_console(tmp_path):
    """Same ASCII-safety contract as `staleness_notification`'s sibling test
    above, for the unverifiable row. Named distinctly so it does not shadow
    that test -- two module-level functions sharing a name silently collapse
    to one collected test, and the point here is both rows are covered."""
    from cash.notebook.ipython.cell_executor import unverifiable_notification
    from cash.notebook.staleness import StalenessTracker

    t = StalenessTracker()
    t.note_source("file")
    unverifiable_notification(t)["code"].encode("ascii")
