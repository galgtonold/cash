"""The sub-unit is judged by decide_cacheability, same as a statement."""
import ast

import pytest

from cash.notebook.call_unit import call_site_is_cacheable


def _call(source: str) -> ast.Call:
    return ast.parse(source).body[0].value


def _check(source, *, user_ns=None, stateful=()):
    return call_site_is_cacheable(
        _call(source),
        user_ns=user_ns if user_ns is not None else {},
        annotation=None,
        is_stateful_call=lambda name: name in stateful,
        scan_forbidden=lambda code, ns, tree: [],
    )


def test_a_plain_call_is_cacheable():
    ok, reasons = _check("compute(x)")
    assert ok, reasons


def test_a_call_that_writes_a_file_is_refused():
    ok, reasons = _check("open('audit.log', 'w')")
    assert not ok
    assert any("Side effect" in r for r in reasons)


def test_a_stateful_callee_is_refused():
    ok, reasons = _check("next_seq()", stateful={"next_seq"})
    assert not ok
    assert reasons == ["Calls @stateful function"]


def test_a_forbidden_function_is_refused():
    ok, reasons = call_site_is_cacheable(
        _call("uuid4()"),
        user_ns={},
        annotation=None,
        is_stateful_call=lambda name: False,
        scan_forbidden=lambda code, ns, tree: ["Calls uuid4"],
    )
    assert not ok
    assert reasons == ["Calls uuid4"]


def test_outputs_empty_means_any_mutation_refuses():
    """A call binds nothing, so `pure_mutations = mutated - {}` = all of them.

    Conservative by construction: a sub-unit that mutates anything visible in
    its own AST is declined, where a statement would have absorbed it into its
    own outputs.
    """
    ok, reasons = _check("acc.extend(items)")
    assert not ok
    assert any("In-place mutation" in r for r in reasons)
