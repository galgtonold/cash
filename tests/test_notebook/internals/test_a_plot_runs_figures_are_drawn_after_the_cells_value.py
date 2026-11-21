"""The figures a plot run leaves open at the end of the cell are drawn where
the inline backend draws them: in ``post_execute``, once the cell's value is
shown, just ahead of the backend's flush and after pyplot's own redraw.

Drawn at the end of the cell's statements, a ``Text`` the cell echoes showed
where the draw had placed it, and a figure that failed to draw lost the error
line IPython prints for pyplot's redraw.
"""

from __future__ import annotations

from types import SimpleNamespace

from cash.notebook.ipython.plot_runs import PlotRuns


def _draw_all_if_interactive():
    pass


def flush_figures():
    pass


def _runs(callbacks):
    return PlotRuns(SimpleNamespace(events=SimpleNamespace(callbacks=callbacks)), None, None)


def test_it_runs_once_between_pyplots_redraw_and_the_flush():
    hooked = [_draw_all_if_interactive, flush_figures]
    done = []
    _runs({"post_execute": hooked})._after_the_value(lambda: done.append(list(hooked)), before=flush_figures)

    assert done == []  # not before the cell's value is shown
    assert hooked[0] is _draw_all_if_interactive and hooked[2] is flush_figures

    for callback in list(hooked):
        callback()

    assert done == [[_draw_all_if_interactive, flush_figures]]  # it took itself off as it ran
    assert hooked == [_draw_all_if_interactive, flush_figures]


def test_without_the_flush_hooked_it_runs_last():
    hooked = [_draw_all_if_interactive]
    _runs({"post_execute": hooked})._after_the_value(lambda: None, before=flush_figures)

    assert len(hooked) == 2 and hooked[0] is _draw_all_if_interactive


def test_a_shell_without_events_runs_it_at_once():
    done = []
    PlotRuns(SimpleNamespace(), None, None)._after_the_value(lambda: done.append(1), before=flush_figures)

    assert done == [1]
