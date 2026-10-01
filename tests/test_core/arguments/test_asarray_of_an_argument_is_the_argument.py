"""``np.asarray(a)`` returns ``a`` itself when ``a`` is already an ndarray.

Writing into the result therefore writes into the caller's array, and the
purity report must say so. ``np.asarray`` of something the function built
(a literal) is still the function's own object.
"""

from __future__ import annotations

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


def _kinds(fn):
    return [i.kind for i in PurityAnalyzer().analyze(fn).issues]


def zero_first_through_asarray(a):
    b = np.asarray(a)
    b[0] = 0
    return float(b.sum())


def zero_first_through_asanyarray(a):
    b = np.asanyarray(a)
    b[0] = 0
    return float(b.sum())


def zero_first_of_own_array():
    b = np.asarray([1.0, 2.0])
    b[0] = 0
    return float(b.sum())


def zero_first_of_a_copy(a):
    b = np.array(a)
    b[0] = 0
    return float(b.sum())


@pytest.mark.parametrize("fn", [zero_first_through_asarray, zero_first_through_asanyarray])
def test_a_write_through_asarray_of_an_argument_is_reported(fn):
    assert "scope_mutation" in _kinds(fn)


@pytest.mark.parametrize("fn", [zero_first_of_own_array, zero_first_of_a_copy])
def test_a_write_into_an_array_the_function_made_is_not_reported(fn):
    assert _kinds(fn) == []
