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
    return magics, published


def test_a_fast_statement_publishes_no_progress_badge(progress_probe):
    magics, published = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="x = 1")
    magics._cancel_progress_badge()
    time.sleep(magics._BADGE_MIN_RENDER_INTERVAL * 2)
    assert published == [], f"a fast statement published {published}"


def test_a_slow_statement_publishes_one_badge_naming_itself(progress_probe):
    magics, published = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="slow = f()")
    time.sleep(magics._BADGE_MIN_RENDER_INTERVAL * 2)
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
    magics._cancel_progress_badge()


def test_cancel_after_the_timer_fired_is_harmless(progress_probe):
    """Cancel always runs, whether or not the timer already fired."""
    magics, published = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="x = 1")
    time.sleep(magics._BADGE_MIN_RENDER_INTERVAL * 2)
    magics._cancel_progress_badge()
    magics._cancel_progress_badge()
    assert len(published) == 1


def test_nothing_publishes_after_the_final_badge(progress_probe):
    """The sharp edge: a late timer would overwrite DONE with a stale RUNNING."""
    magics, published = progress_probe
    magics._arm_progress_badge([], display_id="d", step=1, total=2, code="x = 1")
    magics._cancel_progress_badge()
    published.clear()
    time.sleep(magics._BADGE_MIN_RENDER_INTERVAL * 3)
    assert published == [], "a cancelled timer still fired"
