"""A ``frozen=True`` result the caller changes stops being trusted.

Found while stress-testing the decorator: the audit recorded its
baseline at the object's 8th use as an argument and first COMPARED at the 72nd,
so the ordinary shape -- produce it, change it, pass it again -- served the
pre-change answer indefinitely. Measured: a consumer returned 6 where an
uncached run returns 106, five calls running, with no warning.

``docs/decorator.md``: "Other objects are audited: cash re-hashes one at an
occasional use ... and if it has changed ... KEY-FROZEN-MUTATED names the
producer and the object is keyed by its contents from then on."
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash
from cash.backends import InMemoryBackend


@pytest.fixture
def pipeline():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache(frozen=True)
    def train(seed):
        return {"w": [1, 2, 3]}

    @cash.cache
    def score(model, bias):
        return sum(model["w"]) + bias

    return train, score


def test_a_changed_frozen_result_is_not_served_from_its_old_key(pipeline):
    train, score = pipeline
    model = train(1)
    assert score(model, 0) == 6
    model["w"].append(100)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        after = score(model, 0)
    assert after == 106, "the consumer was served the pre-change answer"
    assert any("KEY-FROZEN-MUTATED" in str(w.message) for w in seen), [str(w.message) for w in seen]


def test_an_unchanged_frozen_result_still_hits(pipeline):
    train, score = pipeline
    model = train(1)
    ran = []
    assert [score(model, i % 2) for i in range(20)] == [6, 7] * 10
    assert not ran


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda m: m["w"].append(100), 106),
        (lambda m: m.__setitem__("w", [1, 2, 3, 100]), 106),
        (lambda m: m["w"].pop(), 3),
    ],
)
def test_the_shapes_a_caller_changes(pipeline, mutate, expected):
    train, score = pipeline
    model = train(1)
    assert score(model, 0) == 6
    mutate(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert score(model, 0) == expected
