"""Progress badges are published on the trailing edge, not the leading one.

Today the pre-statement badge publishes once per statement with no throttle: a
ten-cell run publishes 78 badges, 68 of them progress updates.

The obvious fix -- throttle the leading edge -- is wrong, and this file exists
to keep it that way. Skipping the render that says "now running the slow one"
leaves the PREVIOUS statement on screen for the whole of a long one, which is
worse than the traffic. So the render is deferred instead: armed on a timer,
cancelled if the statement finishes first, and published only if it is still
running -- naming the statement that is actually running.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("IPython")


@pytest.fixture
def progress_probe(monkeypatch):
    """A CashMagics whose badge publishes are recorded instead of displayed."""
    from cash.notebook.ipython.magics import CashMagics

    magics = CashMagics.__new__(CashMagics)
    magics._badge_mode = "html"
    magics._debug = False
    magics._last_badge_render_time = 0.0
    magics._BADGE_MIN_RENDER_INTERVAL = 0.05   # keep the tests quick
    magics._progress_timer = None
    magics._progress_lock = threading.Lock()
    magics._progress_generation = 0

    published: list[dict] = []
    lock_violations: list[str] = []

    # `_arm_progress_badge`'s `fire()` calls `_build_badge_html` (outside the
    # lock) and then `_publish_badge_html` (inside it) separately -- not
    # `_render_interactive_badge` as a single call -- so both halves need
    # faking here, or `fire()` would fall through to the real (slow, real-
    # backend-touching) implementations instead of being observed by these
    # tests. The build fake hands the publish fake everything it captured, so
    # `published` ends up with the same shape the old single-fake-render
    # fixture recorded.
    def fake_build(metrics_list, status="DONE", current_step=0, total_steps=0,
                    current_code=None, cell_total_time=None, timing_breakdown=None):
        # This runs INSIDE `fire()`, on the timer thread, standing in for the
        # expensive half of a render -- the whole reason it must run outside
        # `_progress_lock` (see the comment in `fire()`). Record a violation
        # rather than asserting here: an exception raised off the main thread
        # only produces a PytestUnhandledThreadExceptionWarning, not a failed
        # test, so the check has to surface on the main thread (in the test
        # functions below) to actually gate anything.
        if magics._progress_lock.acquire(blocking=False):
            magics._progress_lock.release()
        else:
            lock_violations.append("the build ran while _progress_lock was held")
        return {
            "status": status,
            "current_step": current_step,
            "total_steps": total_steps,
            "current_code": current_code,
        }

    def fake_publish(html, display_id=None, update_existing=True, _from_thread=False,
                     publisher=None):
        published.append({**html, "display_id": display_id})

    magics._build_badge_html = fake_build  # type: ignore[assignment]
    magics._publish_badge_html = fake_publish  # type: ignore[assignment]
    return magics, published, lock_violations


#: How long a positive assertion will wait for the timer thread. Generous on
#: purpose: it is an upper bound on scheduling latency, not a delay anyone pays
#: -- the polls below return as soon as the badge lands, so the fast case is
#: still milliseconds.
_PUBLISH_TIMEOUT_S = 5.0

#: How long a negative assertion waits before concluding nothing fired. Fixed
#: rather than a multiple of the (deliberately tiny) test interval, so widening
#: the confidence does not depend on that interval staying small.
_QUIET_PERIOD_S = 0.5


def _wait_for_publish(published, timeout: float = _PUBLISH_TIMEOUT_S) -> None:
    """Block until a badge is published, or *timeout* elapses.

    Sleeping a fixed multiple of ``_BADGE_MIN_RENDER_INTERVAL`` and then
    asserting looks equivalent and is not: it gives the timer thread a fixed
    slack (50 ms at the test interval) to be scheduled AND to finish. That held
    locally and failed on a contended macOS runner, twice, where the badge had
    simply not fired yet -- a green assertion about a race rather than about the
    behaviour. Waiting for the condition removes the race without weakening
    anything: the assertions after this call are unchanged, and a badge that
    never fires still fails, just five seconds later.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not published:
        time.sleep(0.005)


def test_a_fast_statement_publishes_no_progress_badge(progress_probe):
    magics, published, lock_violations = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="x = 1")
    magics._cancel_progress_badge()
    time.sleep(_QUIET_PERIOD_S)
    assert published == [], f"a fast statement published {published}"
    assert lock_violations == [], lock_violations


def test_a_slow_statement_publishes_one_badge_naming_itself(progress_probe):
    magics, published, lock_violations = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="slow = f()")
    _wait_for_publish(published)
    # Assert BEFORE cancelling: the statement is still (notionally) running at
    # this point, and this is the claim the design makes -- that the badge is
    # on screen DURING the run, not merely that cancelling produces a correct
    # publish log afterwards. A design that only publishes when cancelled
    # (nothing ever shown while the statement is in flight) would pass this
    # test if the assertion ran after `_cancel_progress_badge()`; asserting
    # first is what rules that out.
    assert len(published) == 1, f"expected exactly one progress badge, got {published}"
    assert published[0]["current_code"] == "slow = f()", (
        "the badge named a different statement than the one that was running"
    )
    # This is the lock-scope regression guard: `fire()` must build the HTML
    # OUTSIDE `_progress_lock` (see the comment in `fire()`). Tidying `fire()`
    # into one `with self._progress_lock:` block leaves every assertion above
    # still passing -- only this one catches it.
    assert lock_violations == [], lock_violations
    magics._cancel_progress_badge()


def test_cancel_after_the_timer_fired_is_harmless(progress_probe):
    """Cancel always runs, whether or not the timer already fired."""
    magics, published, lock_violations = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="x = 1")
    _wait_for_publish(published)
    magics._cancel_progress_badge()
    magics._cancel_progress_badge()
    assert len(published) == 1
    assert lock_violations == [], lock_violations


def test_nothing_publishes_after_the_final_badge(progress_probe):
    """The sharp edge: a late timer would overwrite DONE with a stale RUNNING."""
    magics, published, lock_violations = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="x = 1")
    magics._cancel_progress_badge()
    published.clear()
    time.sleep(_QUIET_PERIOD_S)
    assert published == [], "a cancelled timer still fired"
    assert lock_violations == [], lock_violations
