"""A statement the share check refused is refused again without the check
while its key is the same, and asked again after 1, 2, 4, ... runs.

``for p in points: total += p.x`` leaves ``p`` on an item of ``points``, so
the statement is not stored: a restored ``p`` would not be that item. The same
code over the same inputs makes the same outputs, held the same way, so the
check (which looks for every holder of ``p``) paid its cost on each run only
to refuse again. A holder outside the inputs may go away, so the check comes
back on a growing gap; a refusal it skips costs a store, never a wrong value.
"""

from __future__ import annotations

import types

from cash.notebook.statement import processor as processor_module
from cash.notebook.statement.processor import StatementProcessor
from cash.notebook.statement.store import StatementStore
from tests._cell_driver import run_cash_cell


def _memo():
    return types.SimpleNamespace(
        _share_refused={},
        _SHARE_REFUSED_MAX=StatementProcessor._SHARE_REFUSED_MAX,
        _SHARE_REFUSED_BACKOFF=StatementProcessor._SHARE_REFUSED_BACKOFF,
    )


def _run(key="k"):
    return types.SimpleNamespace(cache_key=key, share_unchecked=False, skip_cache=False, metrics={})


def _checked(memo, reason, key="k"):
    """One run: True when the check ran, after which it refuses with *reason*."""
    run = _run(key)
    if StatementProcessor._refused_again(memo, run):
        assert run.skip_cache and run.metrics["uncacheable_reasons"]
        return False
    StatementProcessor._note_share_refusal(memo, run, reason)
    return True


def test_the_check_comes_back_after_a_doubling_gap():
    memo = _memo()
    pattern = [_checked(memo, "p is held by points") for _ in range(16)]
    # Checked, skipped 1, checked, skipped 2, checked, skipped 4, checked, ...
    assert pattern == [True, False, True, False, False, True] + [False] * 4 + [True] + [False] * 5


def test_an_accepted_check_forgets_the_refusal():
    memo = _memo()
    _checked(memo, "refused")
    assert not _checked(memo, "refused")  # skipped
    assert _checked(memo, None)  # checked, now accepted
    assert _checked(memo, None) and memo._share_refused == {}


def test_a_skipped_refusal_keeps_its_reason_and_unchecked_flag():
    memo = _memo()
    first = _run()
    first.share_unchecked = True
    StatementProcessor._note_share_refusal(memo, first, "gave up")
    again = _run()
    assert StatementProcessor._refused_again(memo, again)
    assert again.share_unchecked and again.metrics["uncacheable_reasons"] == ["gave up"]


def test_another_key_is_checked():
    memo = _memo()
    _checked(memo, "refused", key="a")
    assert _checked(memo, "refused", key="b")
    assert _checked(memo, None, key=None) and None not in memo._share_refused


def test_the_memo_is_bounded():
    memo = _memo()
    for i in range(StatementProcessor._SHARE_REFUSED_MAX + 10):
        _checked(memo, "refused", key=str(i))
    assert len(memo._share_refused) == StatementProcessor._SHARE_REFUSED_MAX


def test_a_loop_over_records_reruns_right_and_checks_less(cash_magics, monkeypatch):
    m = cash_magics
    calls, runs = [], []
    real = processor_module.unrestorable_output_reason

    def spy(outputs, *args, **kwargs):
        if "p" in outputs:
            calls.append(len(runs))
        return real(outputs, *args, **kwargs)

    monkeypatch.setattr(processor_module, "unrestorable_output_reason", spy)
    # The loop takes a few milliseconds: too cheap to store, it would never
    # reach the check (`gets_no_entry`).
    monkeypatch.setattr(StatementStore, "_too_cheap_to_store", lambda *a, **k: False)
    run_cash_cell(m, "class P:\n    def __init__(s, i):\n        s.x = i\n")
    run_cash_cell(m, "points = [P(i) for i in range(3000)]")
    for _ in range(8):
        runs.append(1)
        m.shell.user_ns["total"] = -1
        run_cash_cell(m, "total = 0\nfor p in points:\n    total += p.x\n")
        assert m.shell.user_ns["total"] == sum(range(3000))
        assert m.shell.user_ns["p"] is m.shell.user_ns["points"][-1]
    # Checked on runs 1, 3 and 6: skipped once, then twice.
    assert sorted(set(calls)) == [1, 3, 6]
