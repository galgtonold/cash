"""The seed watch tells a statement that seeded a stream before drawing from
it from one that read the stream first.

Only the first kind may leave the stream out of its key: what it draws
comes from its own seed. A draw, a ``getstate`` or ``setstate`` before the
seed, a ``seed(None)``, or a ``setstate`` after the seed (the stream put
back where it began) all make it a read. So does a ``seed(None)`` after the
seeded draws: the stream then ends at fresh entropy, which the watch also
reports (``streams_seeded_from_entropy``).
"""

from __future__ import annotations

import random

import pytest

from cash.tracking.randomness import (
    capture_rng_state,
    close_seed_watch,
    restore_rng_state,
    streams_seeded_from_entropy,
    streams_set_by_the_statement,
    watch_first_seeds,
    watch_seeds,
)


@pytest.fixture(autouse=True)
def _keep_stream():
    watch_seeds()
    state = random.getstate()
    yield
    random.setstate(state)


def _watched(body) -> frozenset[str]:
    start = capture_rng_state()
    watch_first_seeds(start)
    body()
    return close_seed_watch(start)


def test_a_seed_then_draws_is_seeded_first():
    assert _watched(lambda: (random.seed(3), random.random(), random.seed(4), random.random())) == {"random"}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(lambda: (random.random(), random.seed(3), random.random()), id="draw-first"),
        pytest.param(lambda: (random.getstate(), random.seed(3), random.random()), id="getstate-first"),
        pytest.param(lambda: (random.setstate(random.getstate()), random.seed(3)), id="setstate-first"),
        pytest.param(lambda: (random.seed(None), random.random()), id="entropy-seed"),
        pytest.param(
            lambda: (s := random.getstate(), random.seed(3), random.random(), random.setstate(s)), id="put-back"
        ),
        pytest.param(lambda: (random.seed(3), random.setstate(random.getstate())), id="setstate-after-seed"),
        pytest.param(lambda: random.random(), id="no-seed"),
        pytest.param(lambda: (random.seed(3), random.random(), random.seed()), id="entropy-seed-after"),
    ],
)
def test_a_read_of_the_stream_is_not_seeded_first(body):
    assert _watched(body) == frozenset()


def test_a_watch_opened_inside_another_leaves_the_outer_one_reporting_nothing():
    outer = capture_rng_state()
    watch_first_seeds(outer)
    inner = capture_rng_state()
    watch_first_seeds(inner)
    random.seed(3)
    assert close_seed_watch(inner) == {"random"}
    assert close_seed_watch(outer) == frozenset()


def test_cash_reading_the_stream_is_not_the_statement_reading_it():
    def body():
        capture_rng_state()
        random.seed(3)

    assert _watched(body) == {"random"}


def _set_by_the_statement(body) -> frozenset[str]:
    start = capture_rng_state()
    watch_first_seeds(start)
    body()
    try:
        return streams_set_by_the_statement(start)
    finally:
        close_seed_watch(start)


def test_a_statement_that_puts_the_stream_back_set_it_itself():
    def body():
        s = random.getstate()
        random.seed(3)
        random.random()
        random.setstate(s)

    assert _set_by_the_statement(body) == {"random"}


def test_cash_putting_the_stream_back_is_not_the_statement_setting_it():
    assert _set_by_the_statement(lambda: restore_rng_state(capture_rng_state())) == frozenset()


@pytest.mark.parametrize(
    ("body", "entropy"),
    [
        pytest.param(lambda: (random.seed(3), random.random(), random.seed()), {"random"}, id="after-draws"),
        pytest.param(lambda: (random.seed(None), random.random()), {"random"}, id="first"),
        pytest.param(lambda: (random.seed(3), random.random()), set(), id="a-value"),
    ],
)
def test_a_seed_from_entropy_anywhere_is_reported(body, entropy):
    start = capture_rng_state()
    watch_first_seeds(start)
    body()
    found = streams_seeded_from_entropy(start)
    close_seed_watch(start)
    assert found == entropy
