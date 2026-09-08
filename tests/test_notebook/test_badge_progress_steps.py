"""The progress counter never names a statement that has not started.

Two sources publish it. `_arm_progress_badge` announces the statement that is
running, on a timer, so only a slow one gets named. The post-statement render
fires the moment a statement finishes -- and used to announce the NEXT one,
which at that instant had not begun.

At full speed the throttle drops those renders when they arrive in a burst, so
the wrong number is invisible; on a throttled box (Binder) a render is slow
enough to escape the throttle and the wrong number is what you read. Measured
there on the feature tour's shared-simulation cell -- three trivial assignments
then an eight-second one -- the badge showed `(2/6)` while nothing was running,
and `(7/6)` on the last statement, a step past the end of the cell.

These tests drive the real statement loop with the execution stubbed out, so
they gate the arithmetic rather than the timing.
"""
from __future__ import annotations

import ast

import pytest

pytest.importorskip("IPython")

CELL = "a = 1\nb = 2\nc = 3\nd = slow()\ne = d + 1\nprint(e)\n"


class _RecordingMagics:
    """Stands in for CashMagics, recording every step the badge is told about."""

    _badge_mode = "html"
    _global_ttl = None

    def __init__(self) -> None:
        self.armed: list[tuple[int, int, str | None]] = []
        self.reported: list[tuple[int, int]] = []

    def _arm_progress_badge(self, metrics, display_id, step, total, code):
        self.armed.append((step, total, code))

    def _maybe_progress_badge(self, metrics, display_id, step, total, code):
        self.reported.append((step, total))

    def _cancel_progress_badge(self):
        pass


@pytest.fixture
def executor():
    """A CellExecutor wired to a recording magics, with execution stubbed out.

    `__new__` rather than the constructor: the loop under test needs only the
    badge hooks and `_process_regular_stmt`, and building a real executor would
    drag in a shell, a cache backend and a control-structure processor that
    none of these assertions touch.
    """
    from cash.notebook.ipython.cell_executor import CellExecutor

    ex = CellExecutor.__new__(CellExecutor)
    ex._magics = _RecordingMagics()
    ex._debug = False

    def _stub_process(stmt_code, annotation, occ, is_last, all_metrics,
                      buffered_result_outputs, display_code=None, exec_source=None):
        all_metrics.append({'code': stmt_code, 'status': 'COMPUTED'})
        return buffered_result_outputs

    ex._process_regular_stmt = _stub_process  # type: ignore[assignment]
    return ex


def _run(executor, cell: str = CELL):
    executor._execute_cell_statements(
        cell, ast.parse(cell), [], "display-1", 0.0, {},
    )
    return executor._magics


def test_the_counter_never_runs_past_the_end_of_the_cell(executor):
    magics = _run(executor)
    total = len(ast.parse(CELL).body)
    assert magics.reported, "the post-statement render never fired"
    overshoot = [(s, t) for s, t in magics.reported if s > t]
    assert not overshoot, f"progress badge reported a step past the total: {overshoot}"


def test_a_finished_statement_reports_itself_not_the_next_one(executor):
    """`(2/6)` after statement 1 claims statement 2 is running. It is not."""
    magics = _run(executor)
    total = len(ast.parse(CELL).body)
    assert magics.reported == [(i, total) for i in range(1, total + 1)]


def test_both_sources_agree_on_the_step_a_statement_owns(executor):
    """The armed step and the post-statement step must name the same statement.

    They leapfrogged before: the arm said "running 4", the render that landed
    right after said "running 5". Whichever one the throttle let through was
    the number on screen, so the badge disagreed with itself run to run.
    """
    magics = _run(executor)
    assert [s for s, _t, _c in magics.armed] == [s for s, _t in magics.reported]


def test_the_armed_step_still_names_the_statement_it_belongs_to(executor):
    """Guards the control arm: the fix must not shift what `arm` publishes."""
    magics = _run(executor)
    codes = [ast.unparse(n) for n in ast.parse(CELL).body]
    assert [(s, c) for s, _t, c in magics.armed] == list(enumerate(codes, start=1))
