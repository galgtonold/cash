"""Work counts of the decorator: what a hit and a miss hash, pickle and copy.

A hit should hash its arguments once and copy its value in a few C-level
steps; a miss should pickle its value about once. Each test counts that work
and asserts it under a generous bound, so growth (a second hash of an
argument, a pickle per hit, a deepcopy walk per item) fails here without a
timer. Today's counts are in the comments. Counters: ``tests/_work_counts.py``.
"""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from tests._work_counts import deepcopy_steps, hashed_bytes, pickled_bytes

pytestmark = pytest.mark.core

KB = 1024


def small_dict(n):
    return {f"key{i}": i * 0.5 + n for i in range(10)}


def nested_records(n):
    return [{"id": i + n, "user": f"u{i % 50}", "tags": [i, i + 1], "meta": {"ok": i % 2 == 0}} for i in range(2000)]


def numpy_array(n):
    return np.arange(200_000, dtype=np.float64) + n


def pandas_frame(n):
    return pd.DataFrame({"id": np.arange(20_000) + n, "user": [f"u{i % 100}" for i in range(20_000)]})


@dataclasses.dataclass
class Point:
    x: float
    label: str


def dataclass_list(n):
    return [Point(i * 0.5 + n, f"p{i % 7}") for i in range(2000)]


def keyed_args(arr, frame, config, names):
    return float(arr[:10].sum()) + len(frame) * config["lr"] + len(names)


VALUES = [small_dict, nested_records, numpy_array, pandas_frame, dataclass_list]


def _pickle_size(value) -> int:
    return len(pickle.dumps(value, protocol=5))


@pytest.mark.parametrize("fn", VALUES, ids=lambda fn: fn.__name__)
def test_a_miss_pickles_its_value_about_once(disk_cash, fn):
    f = disk_cash.cache(fn)
    with pickled_bytes() as pickled:
        f(1)
    size = _pickle_size(fn(1))
    # 1.0x today, 2.0x for the dataclass list (stored, and copied into RAM by
    # a pickle round trip); the bound catches one more pickle of the value.
    factor = 2.5 if fn is dataclass_list else 1.5
    assert pickled.bytes >= size, "the value was never pickled: is it stored?"
    assert pickled.bytes <= factor * size + 16 * KB, (pickled.bytes, size)


@pytest.mark.parametrize("fn", VALUES, ids=lambda fn: fn.__name__)
def test_a_hit_hashes_a_small_argument_little_and_copies_in_c(disk_cash, fn):
    f = disk_cash.cache(fn)
    f(1)
    with hashed_bytes() as hashed, pickled_bytes() as pickled, deepcopy_steps() as copies:
        f(1)
    assert f.cache_info()["hits"] == 1
    # ~0.4 KB hashed today (2.4 KB for the dataclass list's class).
    assert hashed.bytes <= 16 * KB
    # No deepcopy walk: JSON-like data, arrays and frames are copied a level
    # or a buffer at a time in C.
    assert copies.calls <= 5
    if fn is dataclass_list:
        # Objects are copied by a pickle round trip (as a disk hit reads
        # them): one pickle of the value, ~1.0x today.
        assert pickled.bytes <= 1.25 * _pickle_size(fn(1)) + 4 * KB
    else:
        assert pickled.bytes <= 4 * KB


def test_a_hit_hashes_a_big_argument_once(disk_cash):
    arr = np.random.default_rng(0).random(1_000_000)
    frame = pd.DataFrame({"a": np.arange(10_000)})
    args = (arr, frame, {"lr": 0.1}, ["a", "b"])
    f = disk_cash.cache(keyed_args)
    f(*args)
    with hashed_bytes() as hashed:
        f(*args)
    assert f.cache_info()["hits"] == 1
    # The argument's content is checked on every hit (it could have changed
    # in place), once: 8.0 MB of an 8 MB array plus a 80 KB frame today.
    assert arr.nbytes <= hashed.bytes <= 1.25 * (arr.nbytes + frame.memory_usage().sum()) + 16 * KB
