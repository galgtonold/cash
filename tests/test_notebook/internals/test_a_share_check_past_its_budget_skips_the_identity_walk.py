"""A share check whose budget the alias check of an output already spent
gives up there, without the identity walk; and the search up from a
frame's column of sessions tells their tuples apart from a frame or a
module by type.

student_8's frame of 87,000 sessions, each a list of ``(action, time)``
records shared with the list of sessions it was built from: the alias
check of each cheap statement on it (a sort, a ``cumcount``, a filter)
read the records past the 0.1 s budget, then the identity walk read them
all again for ids, 0.4 to 1.1 s, only for the share check to give up at
its first step. The search up from the sessions that follows asked a
Python-level type question of each of the 87,000 tuples holding them.

Pins the work, not only the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
import pytest

from cash.notebook import shared_objects as so
from cash.notebook.statement import output_refusals
from tests._cell_driver import run_cash_cell


def _busy(seconds: float) -> None:
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


class _Counted:
    def __init__(self, monkeypatch, check_cost: float) -> None:
        self.walks = 0
        real_coupled, real_identity = output_refusals.identity_coupled_reason, output_refusals.identity_held_reason

        def slow_check(*args, **kwargs):
            # One of the checks before the identity walk, as slow as the
            # alias check of student_8's sessions.
            _busy(check_cost)
            return real_coupled(*args, **kwargs)

        def counted_identity(*args, **kwargs):
            self.walks += 1
            return real_identity(*args, **kwargs)

        monkeypatch.setattr(output_refusals, "identity_coupled_reason", slow_check)
        monkeypatch.setattr(output_refusals, "identity_held_reason", counted_identity)


def _run(cash_magics) -> list:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "rows = [[i, i + 1] for i in range(50)]")
    return [getattr(w.message, "code", None) for w in caught]


@pytest.mark.real_time_limits
def test_a_value_check_past_the_budget_ends_the_check(cash_magics, monkeypatch):
    counted = _Counted(monkeypatch, check_cost=0.3)
    _run(cash_magics)
    assert counted.walks == 0


@pytest.mark.real_time_limits
def test_a_value_check_within_the_budget_goes_on_to_the_identity_walk(cash_magics, monkeypatch):
    counted = _Counted(monkeypatch, check_cost=0.0)
    codes = _run(cash_magics)
    assert counted.walks >= 1
    assert "NOTEBOOK-SHARE-UNCHECKED" not in codes, codes


def test_the_search_up_from_sessions_asks_no_type_of_their_tuples(monkeypatch):
    sessions = [(f"u{i}", [("a", i), ("b", i + 1)]) for i in range(5000)]
    df = pd.DataFrame(sessions, columns=["user", "actions"])
    ns = {"logs": sessions, "df": df}
    asked = [0]
    real = so.instance_of

    def counted(obj, kinds):
        asked[0] += 1
        return real(obj, kinds)

    monkeypatch.setattr(so, "instance_of", counted)
    names, _complete = so.referring_names([df], ns, 30.0, skip_name=lambda name: name == "df")
    assert names == {"logs"}
    assert asked[0] < 1000, asked[0]
