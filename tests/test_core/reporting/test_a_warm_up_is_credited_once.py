"""A hit is credited with what the body would cost now, so a one-off warm-up
paid by a function's first call (a JIT compile, a lazy import) counts at most
once per function per process in ``total_time_saved``, not once per hit."""

from __future__ import annotations

import functools
import logging
import time

from cash import Cash

WARM_UP = 0.4
WARM = 0.05
HITS = 5


@functools.cache
def _warm_up() -> bool:
    """Paid by the first call in the process only, as a JIT compile is."""
    time.sleep(WARM_UP)
    return True


def _decorate(c: Cash):
    @c.cache
    def work(x):
        _warm_up()
        time.sleep(WARM)  # measurable work: the test checks the time a hit saves
        return x

    return work


def test_hits_after_warm_calls_are_credited_the_warm_time(tmp_path):
    _warm_up.cache_clear()
    work = _decorate(Cash(cache_dir=str(tmp_path / "cache")))
    for x in (0, 1, 2):
        work(x)  # the first pays the warm-up; the other two are warm
    for _ in range(HITS):
        work(0)  # the entry the warm-up was stored with
    info = work.cache_info()
    assert info["hits"] == HITS
    # The warm-up was paid by a body in this process: no hit avoided it.
    assert info["total_time_saved"] <= HITS * WARM * 1.6, info["total_time_saved"]
    assert info["total_time_saved"] >= HITS * WARM * 0.9, info["total_time_saved"]


def test_a_fresh_process_credits_the_warm_up_once(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _warm_up.cache_clear()
    first = _decorate(Cash(cache_dir=cache_dir))
    for x in (0, 1):
        first(x)
    _warm_up.cache_clear()  # a new process: nothing warmed up yet
    work = _decorate(Cash(cache_dir=cache_dir))
    work(1)  # a warm entry: its stored time is the warm time
    for _ in range(HITS):
        work(0)
    info = work.cache_info()
    assert info["hits"] == HITS + 1 and info["misses"] == 0
    # The first hit of the warm-up's entry did stand in for the warm-up.
    bound = (HITS + 1) * WARM * 1.6 + WARM_UP * 1.3
    assert info["total_time_saved"] <= bound, info["total_time_saved"]
    assert info["total_time_saved"] >= WARM_UP, info["total_time_saved"]


def test_the_per_call_line_and_the_summary_count_the_same(tmp_path, caplog):
    _warm_up.cache_clear()
    c = Cash(cache_dir=str(tmp_path / "cache"), verbose=True)
    work = _decorate(c)
    for x in (0, 1):
        work(x)
    with caplog.at_level(logging.INFO, logger="cash.calls"):
        work(0)
    hit_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("HIT")]
    assert len(hit_lines) == 1
    saved = work.cache_info()["total_time_saved"]
    assert f"saved {saved:.2f}s" in hit_lines[0], (hit_lines, saved)
    assert saved < WARM_UP / 2
