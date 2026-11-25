"""A check that guards a stored statement, and raises, stores nothing.

Before a statement runs and after it, the processor reads what its entry
must be guarded by: the globals the functions it calls read, the local
modules it reaches, what its imports set on other modules, whether it starts
workers, which modules it sets state on. A check that crashed has not found
"nothing": the statement runs every time and the user is told once, with
``NOTEBOOK-ANALYSIS-FAILED``.
"""

from __future__ import annotations

import logging
import warnings

import pytest

from cash.analysis import cacheability_decision
from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import processor

_PERSIST = CacheAnnotation(persist=True)

#: The check, as the processor names it, and what the user is told it could not do.
_CHECKS = [
    ("called_function_globals", "read the globals the functions it calls read"),
    ("module_globals", "watch the modules it reaches"),
    ("import_state_writes", "read what its imports set on other modules"),
    ("statement_starting_workers", "tell whether it starts worker processes or threads"),
    ("module_state_writes", "tell which modules it sets state on"),
]


@pytest.fixture(autouse=True)
def _report_every_failure(monkeypatch):
    monkeypatch.setattr(cacheability_decision, "_REPORTED_FAILURES", set())
    monkeypatch.setattr(processor, "_LOGGED_ONCE", set())


def _boom(*_args, **_kwargs):
    raise RuntimeError("the check itself failed")


def _run_twice(statement_processor, mock_shell):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statement_processor.process_statement("y = [3, 1, 2]", annotation=_PERSIST)
        first = statement_processor.process_statement("x = sorted(y)", annotation=_PERSIST)
        second = statement_processor.process_statement("x = sorted(y)", annotation=_PERSIST)
    return first, second, [getattr(w.message, "code", None) for w in caught]


def test_the_statement_is_served_when_every_check_answers(statement_processor, mock_shell):
    first, second, codes = _run_twice(statement_processor, mock_shell)
    assert first["status"] == CacheStatus.COMPUTED
    assert second["status"] == CacheStatus.RESTORED
    assert "NOTEBOOK-ANALYSIS-FAILED" not in codes


@pytest.mark.parametrize(("check", "told"), _CHECKS, ids=[name for name, _ in _CHECKS])
def test_a_check_that_raises_runs_the_statement_every_time(statement_processor, mock_shell, monkeypatch, check, told):
    monkeypatch.setattr(processor, check, _boom)
    first, second, codes = _run_twice(statement_processor, mock_shell)
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any(f"cash could not {told}" in reason for reason in second["uncacheable_reasons"])
    assert codes.count("NOTEBOOK-ANALYSIS-FAILED") == 1
    assert mock_shell.user_ns["x"] == [1, 2, 3]


def test_a_failed_watch_of_the_reads_is_logged_once_as_a_warning(statement_processor, mock_shell, monkeypatch, caplog):
    """Failing to watch what a statement changes of the environment is on the
    safe side already (a value no write is noted for counts as changed from
    outside): the statement is stored, and the log says so once."""
    monkeypatch.setattr(processor, "snapshot", _boom)
    with caplog.at_level(logging.DEBUG, logger=processor.logger.name):
        _run_twice(statement_processor, mock_shell)
    said = [r for r in caplog.records if "could not watch the reads" in r.getMessage()]
    assert [r.levelno for r in said][:1] == [logging.WARNING]
    assert all(r.levelno == logging.DEBUG for r in said[1:])


def test_the_walk_inside_the_module_watch_raising_counts_as_the_watch_raising(
    statement_processor, mock_shell, monkeypatch
):
    """The watch of the modules a statement reaches caught its own walk's
    failure and answered "no modules": the processor then had nothing to
    refuse over, and a write to one of those modules would go unseen."""
    from cash.notebook import callee_reach

    real = callee_reach.reached_user_code

    def walk(*args, **kwargs):
        if kwargs.get("close") is False:  # the module watch's own walk
            raise RuntimeError("the walk itself failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(callee_reach, "reached_user_code", walk)
    first, second, codes = _run_twice(statement_processor, mock_shell)
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any("cash could not watch the modules it reaches" in reason for reason in second["uncacheable_reasons"])
    assert mock_shell.user_ns["x"] == [1, 2, 3]
