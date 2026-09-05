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

    published: list[dict] = []

    def fake_render(metrics_list, display_id=None, status="DONE", **kw):
        published.append({"status": status, "display_id": display_id, **kw})

    magics._render_interactive_badge = fake_render  # type: ignore[assignment]
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
    magics._cancel_progress_badge()
    assert len(published) == 1, f"expected exactly one progress badge, got {published}"
    assert published[0]["current_code"] == "slow = f()", (
        "the badge named a different statement than the one that was running"
    )


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
