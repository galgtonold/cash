"""The test-time check that a statement run again says why
(``tests/_miss_reason_audit.py``) flags what it should and nothing else."""

from __future__ import annotations

from cash.notebook.cache_status import CacheStatus
from tests._miss_reason_audit import audit


def _row(code: str = "y = f(x)", status=CacheStatus.COMPUTED, **extra) -> dict:
    return {"code": code, "status": status, **extra}


def test_a_first_run_without_a_reason_is_not_flagged():
    assert audit([_row()], set()) == []


def test_a_repeat_without_a_reason_is_flagged():
    seen: set[str] = set()
    audit([_row()], seen)
    found = audit([_row()], seen)
    assert [f["code"] for f in found] == ["y = f(x)"]


def test_a_repeat_with_any_reason_is_not_flagged():
    seen: set[str] = set()
    audit([_row()], seen)
    for reason in (
        {"miss_reason": "input changed: x"},
        {"uncacheable_reasons": ["it writes a file"]},
        {"skipped_reason": "too big"},
    ):
        assert audit([_row(**reason)], seen) == []


def test_a_served_repeat_is_not_flagged():
    seen: set[str] = set()
    audit([_row()], seen)
    assert audit([_row(status=CacheStatus.RESTORED)], seen) == []


def test_each_pass_of_a_loop_is_its_own_statement():
    """The second pass of a loop's first run is no repeat; the same pass of
    the next run is."""
    seen: set[str] = set()
    first = [_row("t += v", loop_vars={"v": 1}), _row("t += v", loop_vars={"v": 2})]
    assert audit(first, seen) == []
    assert len(audit(first, seen)) == 2


def test_the_loop_markers_in_a_statements_text_are_ignored():
    """An iteration's stored text carries a digest that moves with the
    iterable's lineage: moved, it is still the same statement run again."""
    from cash.control_markers import mark_iteration

    seen: set[str] = set()
    audit([_row(mark_iteration("t += v", "aaa"), loop_vars={"v": 1})], seen)
    assert len(audit([_row(mark_iteration("t += v", "bbb"), loop_vars={"v": 1})], seen)) == 1
