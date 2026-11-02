"""``df.session.argmax()`` leaves ``df`` alone, as ``idxmax`` does.

An unlisted method on a frame is taken to change it, so student_8's
``df.session.argmax()`` moved ``df`` and hashed the 2M-row frame for the
first time (about 6 s). ``argmax`` and ``argmin`` return a position on every
type that has them (pandas, numpy) and never write their receiver.
"""

from __future__ import annotations

import pytest

from cash.analysis.mutations import chain_is_pure


@pytest.mark.parametrize("method", ["argmax", "argmin"])
def test_argmax_and_argmin_are_pure(method):
    assert chain_is_pure(method, frozenset())


def test_a_mutator_inside_the_chain_still_counts():
    assert not chain_is_pure("argmax", frozenset({"pop"}))
