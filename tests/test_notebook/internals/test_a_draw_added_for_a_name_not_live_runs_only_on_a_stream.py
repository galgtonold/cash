"""The replay's gate for a draw the upstream plan added on a name the kernel
did not hold: it runs when the value made again moves as it is read, is
refused when it cannot be run again, and is left out otherwise."""

from __future__ import annotations

import pytest

from cash.notebook.upstream.reexecution_planner import redraw_check, with_conditional_draws
from cash.notebook.upstream.replay import UpstreamRepairGap


def test_a_draw_runs_only_once_its_value_is_a_stream(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("h\n1\n", encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        check = redraw_check({"header = f.readline()": (frozenset({"f"}), False)}, {"f": f})
        assert check("header = f.readline()")
        assert check("other = 1")
    check = redraw_check({"header = f.readline()": (frozenset({"f"}), False)}, {"f": [1, 2]})
    assert not check("header = f.readline()")


def test_a_draw_that_cannot_be_run_again_is_refused_once_its_value_is_a_stream(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("h\n1\n", encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        check = redraw_check({"acc.append(f.readline())": (frozenset({"f"}), True)}, {"f": f})
        with pytest.raises(UpstreamRepairGap):
            check("acc.append(f.readline())")
        assert check("other = 1")
    check = redraw_check({"acc.append(f.readline())": (frozenset({"f"}), True)}, {"f": [1, 2]})
    assert not check("acc.append(f.readline())")


def test_the_draws_go_back_before_what_followed_them_and_leave_the_seeds_their_indices():
    """The random-stream plan never sees a draw that may not run: it gets
    the plan without them, puts its seeds in, and the draws go back before
    the statement that followed each, with no stream index of their own."""
    planned = ["f = open(p)", "header = f.readline()", "n = 1", "tail = f.readline()"]
    chained = ["f = open(p)", "np.random.seed(0)", "n = 1"]
    out, positions = with_conditional_draws(
        planned, chained, {"header = f.readline()": None, "tail = f.readline()": None}
    )
    assert out == ["f = open(p)", "np.random.seed(0)", "header = f.readline()", "n = 1", "tail = f.readline()"]
    assert positions == [0, 1, None, 2, None]
    assert with_conditional_draws(planned, chained, {}) == (chained, [0, 1, 2])
