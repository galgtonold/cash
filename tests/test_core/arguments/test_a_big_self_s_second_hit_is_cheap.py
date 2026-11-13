"""``cash.cache(est.fit)`` on a big estimator: the second hit reads little.

An estimator holds arrays, a frame and a table of records, and its ``fit``
reaches a helper through a classmethod (``Table.build``), as NiMARE's
``_fit`` reaches ``Studyset.from_dataset``. A classmethod read off its class
is a new bound method on every read, so the weak reference the purity
analysis kept to the one it met died at once, the binding check saw it gone,
and every call walked and audited every helper the estimator's code reaches
again: seconds per hit on a real estimator, for a call that took a tenth.

The second hit walks no helper again, reads each array once and the frame
not at all, and a write into any array the estimator holds still misses.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from cash import Cash, FileBackend
from cash.analysis.helper_walk import HelperWalk
from cash.decorator import arg_hashing

pytestmark = pytest.mark.core

N = 200_000


def _scale(values):
    return values * 2.0


class Table:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def build(cls, rows):
        return cls([_scale(r) for r in rows])

    def total(self):
        return float(sum(r.sum() for r in self.rows))


class Estimator:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.coords = rng.random(N)
        self.weights = rng.random(N)
        self.labels = rng.integers(0, 9, N)
        self.frame = pd.DataFrame({"x": rng.random(N), "y": rng.integers(0, 5, N)})
        self.records = [{"id": i, "name": f"s{i}"} for i in range(500)]

    def fit(self, k):
        table = Table.build([self.coords[:k], self.weights[:k], self.labels[:k]])
        return table.total() + float(self.frame["x"].iloc[:k].sum())


@pytest.fixture
def counted(monkeypatch):
    """Helper walks run and content hashes taken, per value type."""
    counts: dict[str, int] = {}
    walk_run = HelperWalk.run

    def run(self):
        counts["walks"] = counts.get("walks", 0) + 1
        return walk_run(self)

    real_hash = arg_hashing.builtin_hash

    @functools.wraps(real_hash)
    def builtin_hash(value, *args, **kwargs):
        name = type(value).__name__
        counts[name] = counts.get(name, 0) + 1
        return real_hash(value, *args, **kwargs)

    monkeypatch.setattr(HelperWalk, "run", run)
    monkeypatch.setattr(arg_hashing, "builtin_hash", builtin_hash)
    return counts


def test_the_second_hit_walks_no_helper_and_reads_each_array_once(tmp_path, counted):
    est = Estimator()
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f = c.cache(est.fit)
    expected = f(1000)
    f(1000)
    counted.clear()
    assert f(1000) == expected
    assert f.cache_info()["hits"] == 2
    assert counted.get("walks", 0) == 0, "a hit walked the helpers of the estimator's code again"
    assert counted.get("ndarray", 0) <= 3, f"a hit read an array more than once: {counted}"
    assert counted.get("DataFrame", 0) == 0, f"a hit read the unchanged frame again: {counted}"


def test_a_new_instance_of_the_same_class_walks_no_helper_again(tmp_path, counted):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    first = c.cache(Estimator().fit)
    expected = first(1000)
    counted.clear()
    second = c.cache(Estimator().fit)
    assert second(1000) == expected
    assert second.cache_info()["hits"] == 1
    assert counted.get("walks", 0) == 0, "the code a class reaches was analysed again for a new instance"


@pytest.mark.parametrize("attr", ["coords", "weights", "labels"])
def test_a_write_into_any_array_of_self_misses(tmp_path, attr):
    est = Estimator()
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f = c.cache(est.fit)
    before = f(1000)
    f(1000)
    getattr(est, attr)[3] += 7
    after = f(1000)
    assert f.cache_info()["misses"] == 2, f"a write into self.{attr} was served the old result"
    assert after == pytest.approx(before + 14.0)


def test_a_write_into_the_frame_of_self_misses(tmp_path):
    est = Estimator()
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f = c.cache(est.fit)
    before = f(1000)
    f(1000)
    est.frame.loc[3, "x"] += 5.0
    assert f(1000) == pytest.approx(before + 5.0)
    assert f.cache_info()["misses"] == 2


def _triple(values):
    return values * 3.0


def test_rebinding_the_classmethod_the_code_calls_still_misses(tmp_path, monkeypatch):
    est = Estimator()
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f = c.cache(est.fit)
    before = f(1000)
    f(1000)
    monkeypatch.setattr(Table, "build", classmethod(lambda cls, rows: cls([_triple(r) for r in rows])))
    after = f(1000)
    assert f.cache_info()["misses"] == 2, "a classmethod rebound on its class was served the old result"
    assert after > before
