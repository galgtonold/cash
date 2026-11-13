"""A statement binding a list of more library objects than the identity
walk looks through (``deltas`` of 1.3 million pandas Timedeltas, a loop
over a grouped frame, student_2) is refused with NOTEBOOK-IDENTITY-HELD,
and the walk saying so stops at its budget. It used to open every item
past it, then the share check's retry with anchors walked the whole list
again though the value had refused on its own: 14 s of a 25 s finalize.
Each object of a kind the walk does not look inside is told by its type
after the first. Pins the work, not the behaviour: a refactor may expect
this to fail.
"""

from __future__ import annotations

import warnings

from cash.notebook.statement import identity_holders, output_refusals
from cash.notebook.statement.identity_holders import _MAX_SEEN, _Walk, identity_held_reason
from tests._cell_driver import run_cash_cell

#: Past the walk's budget by half.
N = _MAX_SEEN * 3 // 2


class _Counted:
    def __init__(self, monkeypatch) -> None:
        self.opens = 0
        self.library_parts = 0
        self.reasons = 0
        real_open, real_parts, real_reason = (
            _Walk._open,
            identity_holders.library_parts,
            (output_refusals.unrestorable_output_reason),
        )

        def counted_open(walk, item, into):
            self.opens += 1
            return real_open(walk, item, into)

        def counted_parts(value):
            self.library_parts += 1
            return real_parts(value)

        def counted_reason(*args, **kwargs):
            self.reasons += 1
            return real_reason(*args, **kwargs)

        monkeypatch.setattr(_Walk, "_open", counted_open)
        monkeypatch.setattr(identity_holders, "library_parts", counted_parts)
        monkeypatch.setattr(output_refusals, "unrestorable_output_reason", counted_reason)


def _library_scalars(n: int) -> list:
    from fractions import Fraction

    assert Fraction not in identity_holders._LEAVES  # a library object, not a leaf
    return [Fraction(i, 7) for i in range(n)]


def test_the_walk_stops_at_its_budget(monkeypatch):
    counted = _Counted(monkeypatch)
    held = identity_held_reason("deltas", _library_scalars(N), {})
    assert held is not None and "more objects than cash looks through" in held
    assert counted.opens <= _MAX_SEEN + 2


def test_a_library_kind_is_asked_for_its_parts_once_per_walk(monkeypatch):
    counted = _Counted(monkeypatch)
    assert identity_held_reason("deltas", _library_scalars(5_000), {}) is None
    assert counted.library_parts == 1


def test_the_statement_is_refused_with_a_warning_after_one_check(cash_magics, cash_instance, monkeypatch):
    # Stored however quick it is, so the share check is asked at all.
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    run_cash_cell(cash_magics, f"from fractions import Fraction\nN = {N}")
    from cash.notebook.statement import processor

    monkeypatch.setattr(processor, "share_check_budget", lambda cost: 600.0)
    # The share check's time spent by the first `_MAX_SEEN` objects: within
    # it, the walk would go on and read the list whole.
    monkeypatch.setattr(identity_holders, "walk_time_left", lambda: False)
    counted = _Counted(monkeypatch)
    monkeypatch.setattr(processor, "unrestorable_output_reason", output_refusals.unrestorable_output_reason)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "deltas = [Fraction(i, 7) for i in range(N)]")
    assert any("NOTEBOOK-IDENTITY-HELD" in str(w.message) for w in caught)
    assert counted.reasons == 1
    assert counted.opens <= _MAX_SEEN + 2
