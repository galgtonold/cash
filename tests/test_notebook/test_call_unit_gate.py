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


def _compute(v):
    return v


def test_variable_lineage_supplied_and_tracked_is_cacheable():
    """Every free name is present in both user_ns and variable_lineage, so
    the missing-lineage reason source is asked for real (variable_lineage is
    supplied, not omitted) and finds nothing to object to.
    """
    ok, reasons = call_site_is_cacheable(
        _call("compute(x)"),
        user_ns={"compute": _compute, "x": 5},
        annotation=None,
        is_stateful_call=lambda name: False,
        scan_forbidden=lambda code, ns, tree: [],
        variable_lineage={"compute": "hash-compute", "x": "hash-x"},
    )
    assert ok, reasons


def test_variable_lineage_supplied_but_missing_an_input_refuses():
    """`x` is a live, non-exempt value present in user_ns but absent from
    variable_lineage -- with variable_lineage actually supplied, the
    missing-lineage reason source must fire for real, not merely be wired up
    and never reached.
    """
    ok, reasons = call_site_is_cacheable(
        _call("compute(x)"),
        user_ns={"compute": _compute, "x": 5},
        annotation=None,
        is_stateful_call=lambda name: False,
        scan_forbidden=lambda code, ns, tree: [],
        variable_lineage={"compute": "hash-compute"},  # x missing on purpose
    )
    assert not ok
    assert reasons == ["Input variable missing lineage"]


def test_omitting_variable_lineage_never_asks_the_missing_lineage_question():
    """Same call, same user_ns as the refusal directly above -- but with
    variable_lineage omitted, the missing-lineage source is not asked at all,
    rather than answered "nothing is missing" by a fabricated always-tracked
    mapping. This is the asymmetry that motivates gating `inputs` itself
    (``inputs = _names_read(call_node) if variable_lineage is not None else
    set()``): the AST-only half decided at rewrite time has no lineage table
    to consult yet, and a caller who "simplified" that back into a sentinel
    mapping passed everywhere would make this test fail exactly like the one
    above, on the same inputs.
    """
    ok, reasons = call_site_is_cacheable(
        _call("compute(x)"),
        user_ns={"compute": _compute, "x": 5},
        annotation=None,
        is_stateful_call=lambda name: False,
        scan_forbidden=lambda code, ns, tree: [],
    )
    assert ok, reasons
