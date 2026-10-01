"""A file write is seen however many user calls down it sits.

``build_deck()`` in a cell calls a helper, which calls a helper, ... and the
fifth one down saves the chart. Calls were followed three deep, so the
statement was cached, and a Restart & Run All served it without the write:
the chart a later cell expected was not there. Every user function a call
reaches is now followed, with a seen set to end call cycles.
"""

from __future__ import annotations

import warnings

from cash.analysis import namespace_effects
from cash.analysis.namespace_effects import statement_user_writer_call, user_callee_writing_files


def save_chart(path):
    open(path, "w", encoding="utf-8").write("chart")


def step_5(path):
    save_chart(path)


def step_4(path):
    step_5(path)


def step_3(path):
    step_4(path)


def step_2(path):
    step_3(path)


def step_1(path):
    step_2(path)


def build_deck(path):
    step_1(path)


def tidy(values):
    return sorted(values)


def summarise(values):
    return len(tidy(values))


def helper(path):
    tidy([path])


def report(path):
    helper(path)


def quiet_3(values):
    return tidy(values)


def quiet_2(values):
    return quiet_3(values)


def quiet_1(values):
    return quiet_2(values)


def loop_a(n, path):
    if n:
        loop_b(n - 1, path)


def loop_b(n, path):
    if n:
        loop_a(n - 1, path)


def test_a_write_six_calls_down_is_seen():
    assert user_callee_writing_files(build_deck) == "save_chart"
    ns = {"build_deck": build_deck}
    assert statement_user_writer_call("build_deck('deck.png')", ns) == ("build_deck", "save_chart")


def test_a_function_that_writes_nothing_however_deep_is_still_not_a_writer():
    assert user_callee_writing_files(summarise) is None
    assert user_callee_writing_files(loop_a) is None, "a call cycle must end, and it writes nothing"


def test_a_rebound_helper_is_judged_by_what_it_is_now(monkeypatch):
    """``report`` calls ``helper``. Asked once while ``helper`` wrote nothing,
    the answer "no" was memoised with ``report``'s own source, and after
    ``helper`` was rebound to a writer the same "no" came back."""
    assert user_callee_writing_files(report) is None
    monkeypatch.setitem(report.__globals__, "helper", build_deck)
    assert user_callee_writing_files(report) == "save_chart"


def test_a_walk_that_runs_away_is_taken_for_a_writer_with_a_warning(monkeypatch):
    """Past the bound that stops a runaway walk, cash cannot tell, so the
    statement runs every time -- and says why."""
    monkeypatch.setattr(namespace_effects, "_CALLEE_LIMIT", 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = user_callee_writing_files(quiet_1)
    assert found, "a walk that was cut short answered 'writes nothing'"
    codes = [getattr(w.message, "code", None) for w in caught]
    assert "KEY-HELPERS-UNWALKABLE" in codes
