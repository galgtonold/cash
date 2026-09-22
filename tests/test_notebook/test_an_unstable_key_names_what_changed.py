""""unstable key" says what kept changing.

Round 29, r29s1: ``NOT CACHED: pings = fleet.load_days(DATA, DAYS) (17.06s) -
unstable key`` after five edits to their helper module. None of the causes
the docs list applied, the docs' advice pointed at nothing, and the badge
did not say that the helper module was what changed each run. The guard
already compares one run's key with the last; it now also remembers which of
the statement's inputs changed between them, and the badge names them.
"""
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.statement.miss_guard import (
    GUARD_AFTER_CONSECUTIVE_CHURN_MISSES,
    GUARD_SKIP_REASON,
    MissGuard,
)


def _churn(guard, changing: str, runs: int = GUARD_AFTER_CONSECUTIVE_CHURN_MISSES + 1):
    for n in range(runs):
        guard.observe("src", f"key{n}", hit=False,
                      components={"DATA": "d1", changing: f"lineage{n}"})


def test_the_guard_names_the_input_that_kept_changing():
    guard = MissGuard(None)
    _churn(guard, "fleet")
    assert guard.is_guarded("src")
    assert guard.cause("src") == "`fleet` changed each run"


def test_a_key_that_changed_with_no_input_changing_says_so():
    guard = MissGuard(None)
    for n in range(GUARD_AFTER_CONSECUTIVE_CHURN_MISSES + 1):
        guard.observe("src", f"key{n}", hit=False, components={"DATA": "d1"})
    assert "outside its inputs" in guard.cause("src")


def test_the_badge_row_shows_the_cause():
    metrics = [{"status": "COMPUTED", "code": "pings = fleet.load_days(DATA, DAYS)",
                "execution_time": 17.0, "total_time": 17.0,
                "skipped_reason": GUARD_SKIP_REASON, "guard_cause": "`fleet` changed each run"}]
    out = render_text(build_interactive_badge(metrics))
    assert "unstable key" in out and "`fleet` changed each run" in out, out

