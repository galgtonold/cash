"""The progress badge, published from a timer thread while a statement runs,
publishes inside cash's own work (`own_work`): what the publisher runs is no
part of the statement's trace. A headless driver's publisher, a function of
its own ``__main__``, was recorded as code every slow statement ran, could
not be found again after a restart, and refused each such entry."""

from __future__ import annotations

import time

import pytest

from cash._own_work import inside

pytest.importorskip("IPython")


def test_the_badge_is_published_inside_own_work():
    from cash.notebook.ipython.badges import BadgePresenter

    badges = BadgePresenter(shell=None, cash=None)
    badges.MIN_RENDER_INTERVAL = 0.01
    seen: list[bool] = []
    badges._build_html = lambda *a, **k: "<b>running</b>"  # type: ignore[assignment]
    badges._publish_html = lambda *a, **k: seen.append(inside())  # type: ignore[assignment]

    badges.arm_progress([], display_id="d", step=1, total=2, code="slow = f()")
    deadline = time.monotonic() + 5.0
    while not seen and time.monotonic() < deadline:
        time.sleep(0.005)
    badges.cancel_progress()

    assert seen == [True]
