"""Unit tests for ``decide_cacheability``.

Tests drive the merge function directly with synthetic inputs — no
``StatementProcessor`` is constructed.  This is the test-surface gain
that motivated extracting the function out of ``process_statement``.
"""
from __future__ import annotations

import ast

import pytest

from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cacheability import analyze_statement
from cash.notebook.cacheability_decision import decide_cacheability


def _analysis(code: str):
    tree = ast.parse(code)
    return tree, analyze_statement(code, tree)


def _decide(
    code: str,
    *,
    inputs: set[str] | None = None,
    outputs: set[str] | None = None,
    annotation: CacheAnnotation | None = None,
    user_ns: dict | None = None,
    variable_lineage: dict | None = None,
    is_stateful_call=lambda _name: False,
    scan_forbidden=lambda _code, _user_ns, _tree: [],
):
    tree, analysis = _analysis(code)
    return decide_cacheability(
        code=code,
        tree=tree,
        inputs=inputs or set(),
        outputs=outputs or set(),
        annotation=annotation,
        analysis=analysis,
        user_ns=user_ns or {},
        variable_lineage=variable_lineage or {},
        is_stateful_call=is_stateful_call,
        scan_forbidden=scan_forbidden,
    )


class TestNoReasonsCacheable:
    def test_pure_assignment_is_cacheable(self):
        cacheable, reasons = _decide("y = x + 1", inputs={"x"}, outputs={"y"},
                                      user_ns={"x": 1}, variable_lineage={"x": "h"})
        assert cacheable is True
        assert reasons == []


class TestAnnotationNoCache:
    def test_no_cache_annotation_short_circuits(self):
        cacheable, reasons = _decide(
            "y = x + 1",
            inputs={"x"}, outputs={"y"},
            annotation=CacheAnnotation(no_cache=True),
            user_ns={"x": 1}, variable_lineage={"x": "h"},
        )
        assert cacheable is False
        assert reasons == ['@cash:no-cache annotation']

    def test_no_cache_wins_over_other_reasons(self):
        """no_cache fires first; mutation reason never gets computed."""
        cacheable, reasons = _decide(
            "lst.append(1)",
            inputs={"lst"}, outputs=set(),
            annotation=CacheAnnotation(no_cache=True),
            user_ns={"lst": []}, variable_lineage={"lst": "h"},
        )
        assert cacheable is False
        assert reasons == ['@cash:no-cache annotation']

    def test_persist_annotation_does_not_skip(self):
        cacheable, _ = _decide(
            "y = x + 1",
            inputs={"x"}, outputs={"y"},
            annotation=CacheAnnotation(persist=True),
            user_ns={"x": 1}, variable_lineage={"x": "h"},
        )
        assert cacheable is True


class TestForbiddenFunctions:
    def test_forbidden_reasons_returned(self):
        def fake_scan(code, user_ns, tree):
            return ["Calls forbidden function: input"]
        cacheable, reasons = _decide(
            "x = input()", inputs=set(), outputs={"x"},
            scan_forbidden=fake_scan,
        )
        assert cacheable is False
        assert reasons == ["Calls forbidden function: input"]

    def test_forbidden_scan_exception_does_not_crash(self):
        def bad_scan(*_a, **_k):
            raise TypeError("boom")
        cacheable, reasons = _decide(
            "y = x + 1", inputs={"x"}, outputs={"y"},
            user_ns={"x": 1}, variable_lineage={"x": "h"},
            scan_forbidden=bad_scan,
        )
        assert cacheable is True
        assert reasons == []


class TestStatefulCall:
    def test_stateful_call_short_circuits(self):
        cacheable, reasons = _decide(
            "y = step()", inputs=set(), outputs={"y"},
            is_stateful_call=lambda name: name == "step",
        )
        assert cacheable is False
        assert reasons == ["Calls @stateful function"]

    def test_first_stateful_match_wins(self):
        """Function returns as soon as one stateful call is found."""
        seen = []
        def lookup(name):
            seen.append(name)
            return name == "step"
        _decide(
            "y = step()", inputs=set(), outputs={"y"},
            is_stateful_call=lookup,
        )
        assert seen == ["step"]

    def test_stateful_lookup_exception_does_not_crash(self):
        def bad_lookup(_name):
            raise AttributeError("boom")
        cacheable, reasons = _decide(
            "y = foo()", inputs=set(), outputs={"y"},
            is_stateful_call=bad_lookup,
        )
        assert cacheable is True
        assert reasons == []


class TestMutationsAndSideEffects:
    def test_in_place_mutation_blocks_caching(self):
        cacheable, reasons = _decide(
            "lst.append(1)", inputs={"lst"}, outputs=set(),
            user_ns={"lst": []}, variable_lineage={"lst": "h"},
        )
        assert cacheable is False
        assert any("In-place mutation" in r for r in reasons)

    def test_mutation_on_output_is_fine(self):
        cacheable, _ = _decide(
            "lst = []\nlst.append(1)",
            inputs=set(), outputs={"lst"},
        )
        assert cacheable is True


class TestMissingLineage:
    def test_input_without_lineage_blocks_caching(self):
        cacheable, reasons = _decide(
            "y = x + 1", inputs={"x"}, outputs={"y"},
            user_ns={"x": 1}, variable_lineage={},  # no lineage for x
        )
        assert cacheable is False
        assert reasons == ['Input variable missing lineage']

    def test_input_not_in_user_ns_blocks_caching(self):
        cacheable, reasons = _decide(
            "y = x + 1", inputs={"x"}, outputs={"y"},
            user_ns={}, variable_lineage={},
        )
        assert cacheable is False
        assert reasons == ['Input variable missing lineage']

    def test_builtin_input_does_not_need_lineage(self):
        cacheable, _ = _decide(
            "y = len(z)", inputs={"len", "z"}, outputs={"y"},
            user_ns={"z": []}, variable_lineage={"z": "h"},
        )
        assert cacheable is True

    def test_get_ipython_skipped(self):
        cacheable, _ = _decide(
            "y = 1", inputs={"get_ipython"}, outputs={"y"},
        )
        assert cacheable is True

    def test_module_input_does_not_need_lineage(self):
        """Module-type inputs are lineage-exempt: their source is captured elsewhere."""
        import os as os_mod
        cacheable, _ = _decide(
            "y = os.getcwd()", inputs={"os"}, outputs={"y"},
            user_ns={"os": os_mod}, variable_lineage={},
        )
        assert cacheable is True

    def test_private_callable_input_does_not_need_lineage(self):
        """Private callables are lineage-exempt (bound methods etc.)."""
        class _Helper:
            def __call__(self): return 1
        cacheable, _ = _decide(
            "y = _h()", inputs={"_h"}, outputs={"y"},
            user_ns={"_h": _Helper()}, variable_lineage={},
        )
        assert cacheable is True


class TestReasonOrdering:
    """The first reason-source that triggers wins; later ones are not consulted."""

    def test_annotation_beats_forbidden(self):
        def scan(*_): return ["forbidden!"]
        _, reasons = _decide(
            "y = x", inputs=set(), outputs={"y"},
            annotation=CacheAnnotation(no_cache=True),
            scan_forbidden=scan,
        )
        assert reasons == ['@cash:no-cache annotation']

    def test_forbidden_beats_stateful(self):
        def scan(*_): return ["forbidden!"]
        _, reasons = _decide(
            "y = step()", inputs=set(), outputs={"y"},
            scan_forbidden=scan,
            is_stateful_call=lambda _: True,
        )
        assert reasons == ["forbidden!"]

    def test_stateful_beats_mutation(self):
        _, reasons = _decide(
            "lst.append(step())", inputs={"lst"}, outputs=set(),
            user_ns={"lst": []}, variable_lineage={"lst": "h"},
            is_stateful_call=lambda name: name == "step",
        )
        assert reasons == ["Calls @stateful function"]

    def test_mutation_beats_missing_lineage(self):
        _, reasons = _decide(
            "lst.append(1)", inputs={"lst"}, outputs=set(),
            user_ns={"lst": []}, variable_lineage={},  # no lineage AND mutation
        )
        assert any("In-place mutation" in r for r in reasons)
        assert 'Input variable missing lineage' not in reasons
