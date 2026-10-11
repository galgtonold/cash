"""A library call asked to write into what it is handed is watched like any
call that may change its argument.

``np.median(a, overwrite_input=True)`` sorts ``a`` part way in place. numpy's
reductions are otherwise known to read their arguments only, so ``a`` was not
fingerprinted around the statement, the statement was cached, and the next
Run All's hit left ``a`` in its old order. ``out=``, ``inplace=`` and any
``overwrite_*=`` that is not the literal ``False`` or ``None`` turn the
read-only rule off, by keyword or by position.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.namespace_effects import bare_call_arguments, call_arguments
from cash.analysis.read_only_callees import reads_its_arguments_only

np = pytest.importorskip("numpy")


def _ns():
    return {"np": np, "a": np.arange(5.0), "b": np.zeros(1), "kw": {"overwrite_input": True}}


@pytest.mark.parametrize(
    "code",
    [
        "np.median(a, overwrite_input=True)",
        "np.percentile(a, 50, overwrite_input=1)",
        "np.quantile(a, 0.5, overwrite_input=flag)",
        "np.median(a, None, None, True)",
        "np.median(a, **kw)",
    ],
)
def test_a_bare_call_asked_to_overwrite_its_input_watches_it(code):
    assert "a" in bare_call_arguments(ast.parse(code), {**_ns(), "flag": True})


@pytest.mark.parametrize(
    "code", ["np.median(a)", "np.median(a, overwrite_input=False)", "np.median(a, None, None, False)"]
)
def test_the_same_call_without_a_write_stays_unwatched(code):
    """The control: the fast path for a reading call is kept."""
    assert bare_call_arguments(ast.parse(code), _ns()) == frozenset()


@pytest.mark.parametrize(
    "code",
    [
        "m = np.median(a, overwrite_input=True)",
        "m = np.median(a, None, None, True)",
        "s = np.mean(a, out=b)",
        "s = np.mean(a, 0, None, b)",
    ],
)
def test_a_kept_result_does_not_hide_a_write(code):
    assert {"a", "b"} & call_arguments(ast.parse(code), _ns())


def test_a_kept_result_of_a_reading_call_stays_unwatched():
    assert call_arguments(ast.parse("m = np.median(a)"), _ns()) == frozenset()
    assert call_arguments(ast.parse("m = np.median(a, overwrite_input=False)"), _ns()) == frozenset()


def test_the_runtime_check_reads_the_values_passed():
    a = np.arange(3.0)
    assert reads_its_arguments_only(np.median, (a,), {})
    assert reads_its_arguments_only(np.median, (a,), {"overwrite_input": False, "out": None})
    assert not reads_its_arguments_only(np.median, (a,), {"overwrite_input": True})
    assert not reads_its_arguments_only(np.mean, (a,), {"out": np.zeros(())})
    assert not reads_its_arguments_only(np.median, (a, None, None, True), {})
    assert not reads_its_arguments_only(np.median, (a,), {}, open_ended=True)
    # A function with no write parameter is read-only however it is called.
    assert reads_its_arguments_only(np.unique, (a,), {}, open_ended=True)
