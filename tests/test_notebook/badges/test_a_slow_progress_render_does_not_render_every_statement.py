"""A progress render slower than the throttle's interval is still throttled.

The throttle allowed one RUNNING render per interval, counted from the start
of the last one. A render draws every row of the cell so far, so in a long
cell on a busy machine one render can take longer than the interval: by the
next statement the interval had passed, that statement rendered every row
again, and so did each one after it. The interval is counted from the end
of the render.
"""

from __future__ import annotations

import pytest

pytest.importorskip("IPython")

from cash.notebook.ipython import badges as badges_module
from cash.notebook.ipython.badges import BadgePresenter


@pytest.fixture
def slow_renders(monkeypatch):
    """A presenter on a clock the test moves, whose render takes 0.5 s of
    it: longer than the 0.3 s interval. Returns ``(presenter, clock, renders)``."""
    clock = [1000.0]
    monkeypatch.setattr(badges_module.time, "time", lambda: clock[0])
    presenter = BadgePresenter(shell=None, cash=None)
    assert presenter.MIN_RENDER_INTERVAL == 0.3
    renders: list[int] = []

    def render(metrics, **kwargs):
        renders.append(kwargs["current_step"])
        clock[0] += 0.5

    monkeypatch.setattr(presenter, "render", render)
    return presenter, clock, renders


def test_statements_right_after_a_slow_render_do_not_render(slow_renders):
    presenter, clock, renders = slow_renders
    for step in range(1, 6):
        presenter.maybe_progress([], "badge", step, 5, f"x{step} = 1")
        clock[0] += 0.01  # the statement itself
    assert renders == [1]


def test_a_render_is_allowed_again_an_interval_after_the_last_one_ended(slow_renders):
    presenter, clock, renders = slow_renders
    presenter.maybe_progress([], "badge", 1, 3, "x1 = 1")
    clock[0] += 0.29
    presenter.maybe_progress([], "badge", 2, 3, "x2 = 1")
    clock[0] += 0.02
    presenter.maybe_progress([], "badge", 3, 3, "x3 = 1")
    assert renders == [1, 3]


def test_a_render_that_raises_still_counts(slow_renders, monkeypatch):
    presenter, clock, renders = slow_renders

    def broken(metrics, **kwargs):
        renders.append(kwargs["current_step"])
        clock[0] += 0.5
        raise RuntimeError("render failed")

    monkeypatch.setattr(presenter, "render", broken)
    with pytest.raises(RuntimeError):
        presenter.maybe_progress([], "badge", 1, 2, "x1 = 1")
    presenter.maybe_progress([], "badge", 2, 2, "x2 = 1")
    assert renders == [1]
