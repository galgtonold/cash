"""Regression for finding #10: ``cache_info()['total_time_saved']`` must credit
the *avoided compute time*, not the cache-lookup time.

Before the fix it summed the hit's own (microsecond) lookup duration, so it
under-reported savings by ~4 orders of magnitude.
"""
from __future__ import annotations

import time

from cash import Cash


def test_total_time_saved_credits_avoided_compute(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def slow(x):
        time.sleep(0.3)
        return x * 2

    slow(1)            # miss (~0.3s compute, stored)
    slow(1)            # hit
    slow(1)            # hit

    info = slow.cache_info()
    assert info["hits"] == 2
    assert info["misses"] == 1
    # Two hits of a ~0.3s function -> ~0.6s saved, not ~0.0 (lookup time).
    assert info["total_time_saved"] > 0.4, info["total_time_saved"]


def test_no_time_saved_on_pure_misses(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def f(x):
        return x

    f(1)
    f(2)
    f(3)               # three distinct args -> all misses
    info = f.cache_info()
    assert info["hits"] == 0
    assert info["total_time_saved"] == 0.0


def test_time_saved_survives_cross_instance_restore(tmp_path):
    """A hit served from disk in a fresh instance credits the stored compute
    time, so the number is meaningful across process restarts."""
    cache_dir = str(tmp_path / "cache")

    c1 = Cash(cache_dir=cache_dir)

    @c1.cache
    def slow(x):
        time.sleep(0.25)
        return x * 2

    slow(7)            # compute + persist

    c2 = Cash(cache_dir=cache_dir)

    @c2.cache
    def slow(x):       # noqa: F811 - same source -> same key space
        time.sleep(0.25)
        return x * 2

    slow(7)            # warm-up (cold-key transition)
    slow(7)            # restored hit
    info = slow.cache_info()
    assert info["hits"] >= 1
    assert info["total_time_saved"] > 0.15, info["total_time_saved"]
