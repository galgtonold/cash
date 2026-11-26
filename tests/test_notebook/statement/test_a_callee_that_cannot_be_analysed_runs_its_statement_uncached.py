"""A statement calling a function cash cannot analyse runs uncached.

What a function of the user's reads from the environment joins the key of
the statement that calls it, and a read of the clock or a fresh UUID in it
keeps the statement from being cached: both come from the analysis of the
function. When that analysis raises, "could not tell" was read as "reads
nothing": a statement returning a fresh UUID was cached and gave the first
one on every run, with no word of it, and the variable was in no key. The
statement now runs every time and says so once
(``NOTEBOOK-ANALYSIS-FAILED``), and what is built on its value follows the
value. ``# @cash:assume-safe`` caches it, as it does for a read cash named.
"""

from __future__ import annotations

import warnings

import pytest

from cash.analysis import cacheability_decision
from cash.analysis.annotations import CacheAnnotation
from cash.notebook import lineage_formula
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

_PERSIST = CacheAnnotation(persist=True)
_WAIVED = CacheAnnotation(persist=True, assume_safe=True)

READER = """
import os
import uuid

def mode():
    return os.environ.get("CASH_T_MODE", "unset")

def token():
    return uuid.uuid4().hex

def other(v):
    return v + 1
"""


@pytest.fixture(autouse=True)
def _report_every_failure(monkeypatch):
    monkeypatch.setattr(cacheability_decision, "_REPORTED_FAILURES", set())


@pytest.fixture
def reader(cash_magics, mock_shell, monkeypatch):
    """A notebook that defined ``mode``, which reads an environment
    variable, and ``token``, which returns a fresh UUID."""
    monkeypatch.setenv("CASH_T_MODE", "a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, READER)
    return mock_shell.user_ns


@pytest.fixture
def unreadable(monkeypatch):
    """The analysis of ``mode`` and of ``token`` raises, as a construct it
    cannot handle would make it."""
    real = lineage_formula.get_analyzer()

    class _Analyzer:
        def analyze(self, fn):
            if getattr(fn, "__name__", "") in ("mode", "token"):
                raise RuntimeError("a construct the analysis cannot read")
            return real.analyze(fn)

    monkeypatch.setattr(lineage_formula, "get_analyzer", _Analyzer)


def _run(processor, code, annotation=_PERSIST):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = processor.process_statement(code, annotation=annotation)
    return result, [getattr(w.message, "code", None) for w in caught]


class TestAnalysed:
    """The controls: with the analysis working, each read is found."""

    def test_the_variable_is_in_the_key(self, reader, statement_processor, monkeypatch):
        first, _ = _run(statement_processor, "m = mode()")
        again, _ = _run(statement_processor, "m = mode()")
        assert first["status"] == CacheStatus.COMPUTED
        assert again["status"] != CacheStatus.COMPUTED, "an unchanged variable must hit"
        monkeypatch.setenv("CASH_T_MODE", "b")
        changed, _ = _run(statement_processor, "m = mode()")
        assert changed["status"] == CacheStatus.COMPUTED
        assert reader["m"] == "b"

    def test_a_fresh_uuid_runs_every_time(self, reader, statement_processor):
        _run(statement_processor, "t = token()")
        first = reader["t"]
        second, _ = _run(statement_processor, "t = token()")
        assert second["status"] == CacheStatus.COMPUTED
        assert reader["t"] != first


def test_a_fresh_uuid_is_not_frozen(reader, unreadable, statement_processor):
    first, codes = _run(statement_processor, "t = token()")
    value = reader["t"]
    assert codes == ["NOTEBOOK-ANALYSIS-FAILED"]
    second, codes = _run(statement_processor, "t = token()")
    assert reader["t"] != value, "the first UUID was served again"
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any("token()" in reason and "RuntimeError" in reason for reason in second["uncacheable_reasons"])
    assert codes == [], "said once"


def test_the_statement_and_what_is_built_on_it_follow_the_variable(
    reader, unreadable, statement_processor, monkeypatch
):
    _run(statement_processor, "m = mode()")
    _run(statement_processor, "u = m.upper()")
    assert (reader["m"], reader["u"]) == ("a", "A")
    monkeypatch.setenv("CASH_T_MODE", "b")
    second, _ = _run(statement_processor, "m = mode()")
    below, _ = _run(statement_processor, "u = m.upper()")
    assert (reader["m"], reader["u"]) == ("b", "B"), "the statement below was served for the old value"
    assert second["status"] == CacheStatus.COMPUTED
    assert any("mode()" in reason for reason in second["uncacheable_reasons"])
    # The value did not change: what is built on it is served.
    _run(statement_processor, "m = mode()")
    below, _ = _run(statement_processor, "u = m.upper()")
    assert below["status"] != CacheStatus.COMPUTED


def test_a_waiver_caches_it(reader, unreadable, statement_processor):
    """``# @cash:assume-safe`` accepts what the function reads, as it does
    for a read cash named: the first UUID is kept."""
    _run(statement_processor, "t = token()", _WAIVED)
    value = reader["t"]
    second, codes = _run(statement_processor, "t = token()", _WAIVED)
    assert second["status"] != CacheStatus.COMPUTED
    assert reader["t"] == value
    assert codes == []


def test_a_statement_calling_only_functions_that_are_analysed_still_caches(reader, unreadable, statement_processor):
    _run(statement_processor, "x = other(1)")
    again, codes = _run(statement_processor, "x = other(1)")
    assert again["status"] != CacheStatus.COMPUTED, again.get("uncacheable_reasons")
    assert codes == []


def test_the_function_that_could_not_be_read_is_named(reader, unreadable):
    failed = lineage_formula.unanalysed_callee("m = mode()", reader)
    assert failed is not None
    fn, error = failed
    assert fn is reader["mode"]
    assert isinstance(error, RuntimeError)
    assert lineage_formula.unanalysed_callee("x = other(1)", reader) is None
    assert lineage_formula.unanalysed_callee("x = 1 + 2", reader) is None


def test_through_a_cell_a_fresh_uuid_is_not_frozen(reader, unreadable, cash_magics):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, "# @cash:persist\nt = token()")
        value = reader["t"]
        run_cash_cell(cash_magics, "# @cash:persist\nt = token()")
    assert reader["t"] != value
