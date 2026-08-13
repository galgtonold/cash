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
