"""A safety check that raises makes its statement run uncached.

The checks that stop a statement from being cached -- the scan for calls it
must not cache, the ``@stateful`` and file-writer checks, the analysis of what
the called functions write -- answer "cannot tell" when they crash. "Cannot
tell" must not read as "pure": the statement runs every time and the user is
told once, with ``NOTEBOOK-ANALYSIS-FAILED``.
"""

from __future__ import annotations

import ast
import warnings

import pytest

from cash.analysis import cacheability_decision
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability
from cash.analysis.callee_effects import callee_global_mutations
from cash.analysis.mutation_effects import statement_effects
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture(autouse=True)
def _report_every_failure(monkeypatch):
    monkeypatch.setattr(cacheability_decision, "_REPORTED_FAILURES", set())


def _boom(*_args, **_kwargs):
    raise RuntimeError("the check itself failed")


def _decide(code, **overrides):
    tree = ast.parse(code)
    kwargs = dict(
        code=code,
        tree=tree,
        inputs=set(),
        outputs={"x"},
        annotation=None,
        analysis=analyze_statement(code, tree),
        user_ns={},
        variable_lineage={},
        is_stateful_call=lambda _name: False,
        scan_forbidden=lambda *_a: [],
    )
    kwargs.update(overrides)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = decide_cacheability(**kwargs)
    return verdict, [getattr(w.message, "code", None) for w in caught]


def test_a_crashed_forbidden_scan_is_not_cacheable():
    (cacheable, reasons), codes = _decide("x = f()", scan_forbidden=_boom)
    assert cacheable is False
    assert "RuntimeError" in reasons[0]
    assert codes == ["NOTEBOOK-ANALYSIS-FAILED"]


def test_a_crashed_stateful_check_is_not_cacheable():
    (cacheable, reasons), _ = _decide("x = f()", is_stateful_call=_boom)
    assert cacheable is False
    assert "could not" in reasons[0]


def test_the_same_failure_is_reported_once():
    _decide("x = f()", scan_forbidden=_boom)
    _, codes = _decide("x = f()", scan_forbidden=_boom)
    assert codes == []


def test_a_callee_without_source_writes_no_global():
    def no_source(_name):
        raise OSError("no source")

    assert callee_global_mutations(ast.parse("f()"), no_source) == frozenset()


def test_a_resolver_that_crashes_is_not_read_as_no_writes():
    with pytest.raises(RuntimeError):
        callee_global_mutations(ast.parse("f()"), _boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = statement_effects("x = f(y)", ast.parse("x = f(y)"), namespace={}, resolve_source=_boom)
    assert effects.unanalysed
    assert effects.inputs >= {"y"} and effects.outputs == {"x"}


def test_the_statement_runs_every_time(statement_processor, mock_shell, monkeypatch):
    mock_shell.user_ns["f"] = lambda v: v + 1
    mock_shell.user_ns["y"] = 1
    monkeypatch.setattr(statement_processor, "resolve_live_function_source", _boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = statement_processor.process_statement("x = f(y)", annotation=_PERSIST)
        second = statement_processor.process_statement("x = f(y)", annotation=_PERSIST)
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any("could not analyse" in r for r in second["uncacheable_reasons"])
