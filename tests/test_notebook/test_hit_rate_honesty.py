"""The hit rate must not understate a session where caching worked (CAS-177).

``%cash_stats`` reported ``Cache hit rate: 14.9%`` for a session in which 100%
of the expensive statements hit. The denominator counted every ``print`` and
``import`` -- statements cash deliberately never tried to cache, because they
sit below its own "too cheap to cache" floor. Counting cash's correct decisions
as misses makes a perfect session look like a failure.

CAS-157 fixed an OVERstatement of savings; this is the same failure inverted.
The binding rule is CAS-157's: the number must not imply a conclusion the data
does not support, in EITHER direction. So these tests pin both sides --

* a session of expensive hits buried in trivia must NOT read as a failure; and
* the fix must not become its own overstatement: a session that genuinely
  missed its expensive statements must still read as a miss, and a session with
  no cacheable statements at all must report no rate rather than a flattering
  (or damning) one.
"""
from __future__ import annotations

import json

import pytest

from cash.notebook.cache_status import CacheStatus


def _trivia(n, code_prefix="print"):
    """Sub-floor statements: cash computes them and declines to cache them."""
    return [{'status': CacheStatus.COMPUTED, 'execution_time': 0.0002,
             'code': f'{code_prefix}({i})'} for i in range(n)]


def _expensive_hits(n, saved=13.5):
    return [{'status': CacheStatus.RESTORED, 'saved_time': saved,
             'code': f'model_{i} = fit()'} for i in range(n)]


def _expensive_misses(n, cost=13.5):
    return [{'status': CacheStatus.COMPUTED, 'execution_time': cost,
             'code': f'model_{i} = fit()'} for i in range(n)]


def test_perfect_session_buried_in_trivia_does_not_read_as_failure(
    cash_magics, capsys,
):
    """The reported session: 7/7 expensive statements hit, 40 trivia alongside."""
    cash_magics._update_session_stats(
        _trivia(40) + _expensive_hits(7), cell_total_time=1.6,
    )
    cash_magics.cash_stats('')
    out = capsys.readouterr().out

    assert '100.0%' in out, "a session where every expensive statement hit"
    assert 'worth caching' in out
    # The arithmetically-true figure is kept, but explained rather than headlined.
    assert '14.9%' in out
    assert 'not misses' in out


def test_trivia_are_excluded_from_the_denominator(cash_magics):
    """Trivia must not enter the cacheable counters at all."""
    cash_magics._update_session_stats(_trivia(40), cell_total_time=0.1)
    stats = cash_magics._session.stats

    assert stats['statements_computed'] == 40
    assert stats['statements_cacheable_hit'] == 0
    assert stats['statements_cacheable_miss'] == 0


def test_a_real_miss_still_reads_as_a_miss(cash_magics, capsys):
    """The fix must not flatter: expensive statements that MISSED count."""
    cash_magics._update_session_stats(
        _trivia(40) + _expensive_misses(7), cell_total_time=95.0,
    )
    cash_magics.cash_stats('')
    out = capsys.readouterr().out

    assert cash_magics._session.stats['statements_cacheable_miss'] == 7
    assert '0.0%' in out, "7 expensive misses and 0 hits must read as 0%"


def test_half_hit_session_reports_the_true_cacheable_rate(cash_magics, capsys):
    cash_magics._update_session_stats(
        _trivia(10) + _expensive_hits(3) + _expensive_misses(1),
        cell_total_time=20.0,
    )
    cash_magics.cash_stats('')
    out = capsys.readouterr().out

    assert '75.0%' in out
    assert '(3/4 statements worth caching)' in out


def test_no_cacheable_statements_reports_no_rate(cash_magics, capsys):
    """Honest silence: a rate over an empty denominator is undefined.

    Printing 0% would blame cash for correctly declining to cache a notebook
    that contained nothing worth caching.
    """
    cash_magics._update_session_stats(_trivia(12), cell_total_time=0.05)
    cash_magics.cash_stats('')
    out = capsys.readouterr().out

    assert 'n/a' in out
    assert 'no statement was expensive enough to cache' in out
    assert '0.0%' not in out


def test_single_number_when_there_is_no_trivia(cash_magics, capsys):
    """With nothing to disambiguate, don't print two rates."""
    cash_magics._update_session_stats(_expensive_hits(4), cell_total_time=1.0)
    cash_magics.cash_stats('')
    out = capsys.readouterr().out

    assert '100.0%  (4/4 statements)' in out
    assert 'counting all' not in out


def test_json_exposes_the_cacheable_rate_and_nulls_it_when_undefined(
    cash_magics, capsys,
):
    cash_magics._update_session_stats(
        _trivia(40) + _expensive_hits(7), cell_total_time=1.6,
    )
    cash_magics.cash_stats('json')
    payload = json.loads(capsys.readouterr().out)

    assert payload['hit_rate_cacheable_percent'] == 100.0
    assert payload['statements_cacheable_total'] == 7
    assert payload['hit_rate_percent'] == 14.9  # unchanged, still available

    # Undefined must serialise as null, never 0.0.
    cash_magics.cash_stats('reset')
    capsys.readouterr()
    cash_magics._update_session_stats(_trivia(5), cell_total_time=0.01)
    cash_magics.cash_stats('json')
    payload = json.loads(capsys.readouterr().out)
    assert payload['hit_rate_cacheable_percent'] is None


def test_reset_clears_the_cacheable_counters(cash_magics, capsys):
    """A reset must forget these too, or the next session inherits them."""
    cash_magics._update_session_stats(
        _trivia(3) + _expensive_hits(2), cell_total_time=1.0,
    )
    cash_magics.cash_stats('reset')
    stats = cash_magics._session.stats

    assert stats['statements_cacheable_hit'] == 0
    assert stats['statements_cacheable_miss'] == 0


def test_reset_zeroes_every_stat_a_session_can_hold(cash_magics, capsys):
    """Pins the structural guarantee, not just today's key list.

    ``reset`` re-listed the keys by hand, so adding a counter to the session
    left it carrying over the reset -- reported as a success. This asserts
    reset == a fresh session for EVERY key, so the next counter added cannot
    reintroduce the drift.
    """
    from cash.notebook.ipython.magics import new_session_stats

    for key in cash_magics._session.stats:
        cash_magics._session.stats[key] = 99
    cash_magics._session.measured_compute['x = f()'] = 1.0

    cash_magics.cash_stats('reset')

    assert cash_magics._session.stats == new_session_stats()
    # CAS-157: the verified-saving baselines are part of the stats, not the
    # cache, so a reset must drop them too.
    assert cash_magics._session.measured_compute == {}


@pytest.mark.parametrize("floor, expect_cacheable", [
    (0.01, 1),   # default floor: a 20ms statement is worth caching
    (5.0, 0),    # raised floor: the same statement is trivia
])
def test_the_split_follows_cashs_own_floor_not_a_second_opinion(
    cash_magics, floor, expect_cacheable,
):
    """"Worth caching" must mean exactly what the cache meant by it."""
    cash_magics._cash_instance.config.min_execution_time_to_cache_seconds = floor

    cash_magics._update_session_stats(
        [{'status': CacheStatus.COMPUTED, 'execution_time': 0.02, 'code': 'x = f()'}],
        cell_total_time=0.02,
    )

    assert cash_magics._session.stats['statements_cacheable_miss'] == expect_cacheable
