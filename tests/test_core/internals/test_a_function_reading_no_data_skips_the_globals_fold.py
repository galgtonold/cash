"""A cached function that reads no module data skips the globals fold on a hit.

Whether a function reads any global, ``module.ATTR``, body import or
docstring is a fact of its code, worked out once per code object. The fold
still ran for the cached function itself on every hit -- building its
reader, its package, its drift state -- to find nothing. A function that
does read a global is still folded, and a default lambda's reads still key.
"""

from __future__ import annotations

import pytest

from cash.decorator.globals_fold import GlobalsFold

RATE = 2


def reads_nothing(i, scale=2.0):
    return {"i": i, "v": i * scale}


def reads_rate(i):
    return i * RATE


def test_a_hit_of_a_function_reading_no_data_runs_no_globals_fold(disk_cash, monkeypatch):
    cached = disk_cash.cache(reads_nothing)
    cached(1)
    folds = []
    real = GlobalsFold._fold_read_globals

    def counting(self, func, *args, **kwargs):
        folds.append(func)
        return real(self, func, *args, **kwargs)

    monkeypatch.setattr(GlobalsFold, "_fold_read_globals", counting)
    for _ in range(5):
        assert cached(1) == {"i": 1, "v": 2.0}
    assert reads_nothing not in folds


def test_a_function_reading_a_global_still_keys_it(disk_cash, monkeypatch):
    cached = disk_cash.cache(reads_rate)
    assert cached(3) == 6
    monkeypatch.setitem(globals(), "RATE", 5)
    assert cached(3) == 15
    assert cached.cache_info()["misses"] == 2
