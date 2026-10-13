"""A ``frozen=True`` frame restored from the cache is keyed by the content
hash stored with it, not read again by every process that restores it.

A pool of workers each restored a 200 MB frozen frame and passed it to a
cached consumer, which hashed all of it before it could look its own entry
up: about as long as the restore, in every worker, on every warm run, while
``docs/decorator.md`` says a frozen result is keyed by its producer "in
microseconds". The content hash is now taken once, when the result is
stored, and comes back with every hit, so the consumer's key is the same
content hash without the read. A restored frame that is changed is still
keyed by what it holds.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash
from cash.backends._writes import all_pending_writes
from tests._work_counts import hashed_bytes

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.core

N = 400_000  # 3.2 MB of float64, plus the labels


def _pipeline(cache_dir):
    """The producer and its consumer, as a fresh process over *cache_dir*
    builds them."""
    c = Cash(cache_dir=str(cache_dir), register_magic=False)

    @c.cache(frozen=True)
    def load(n):
        return pd.DataFrame({"a": np.arange(n, dtype=np.float64), "b": np.arange(n) % 7})

    @c.cache
    def total(df, k):
        return float(df["a"].sum()) * k + len(df)

    return c, load, total


def _settle():
    for queue in all_pending_writes():
        queue.wait_all()


@pytest.fixture
def warm(tmp_path):
    """A cache that holds ``load(N)`` and ``total(load(N), 2)``, written by
    an earlier "process"."""
    _c, load, total = _pipeline(tmp_path / ".cash")
    total(load(N), 2)
    _settle()
    return tmp_path / ".cash"


def test_a_consumer_keys_a_restored_frozen_frame_without_hashing_it(warm):
    _c, load, total = _pipeline(warm)
    df = load(N)
    assert load.cache_info()["hits"] == 1
    with hashed_bytes() as hashed:
        assert total(df, 2) == total.__wrapped__(df, 2)
    assert total.cache_info()["hits"] == 1, "the consumer's key changed"
    # The key of a small call: a few KB today. The frame is 6.4 MB.
    frame_bytes = int(df.memory_usage(deep=True).sum())
    assert hashed.bytes < frame_bytes / 20, f"hashed {hashed.bytes} bytes of a {frame_bytes}-byte frame"


def test_the_consumer_keys_a_computed_and_a_restored_frame_alike(tmp_path):
    """The stored hash is the content hash: a consumer called on the frame
    a miss computed finds the entry a consumer of the restored frame made,
    and the other way round."""
    _c, load, total = _pipeline(tmp_path / ".cash")
    total(load(N), 3)
    _settle()
    _c2, load2, total2 = _pipeline(tmp_path / ".cash")
    assert total2(load2(N), 3) == total.__wrapped__(load.__wrapped__(N), 3)
    assert (load2.cache_info()["hits"], total2.cache_info()["hits"]) == (1, 1)
    # A consumer whose own entry is new still matches what a content hash gives.
    total2(load2(N), 4)
    _settle()
    _c3, load3, total3 = _pipeline(tmp_path / ".cash")
    computed = load3.__wrapped__(N)
    assert total3(computed, 4) == total.__wrapped__(computed, 4)
    assert total3.cache_info()["hits"] == 1


def test_a_restored_frozen_frame_whose_shape_changes_is_keyed_by_its_content(warm):
    _c, load, total = _pipeline(warm)
    df = load(N)
    df.drop(index=df.index[-10:], inplace=True)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = total(df, 2)
    assert got == total.__wrapped__(df, 2), "served the answer for the frame as it was stored"
    assert any("KEY-FROZEN-MUTATED" in str(w.message) for w in seen), [str(w.message) for w in seen]


def test_a_restored_frozen_frame_changed_in_place_is_caught_by_the_audit(warm, monkeypatch):
    """Under CASH_DEBUG every use is audited, against the stored hash."""
    monkeypatch.setenv("CASH_DEBUG", "1")
    _c, load, total = _pipeline(warm)
    df = load(N)
    df.loc[0, "a"] = 1e9
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = total(df, 2)
    assert got == total.__wrapped__(df, 2), "served the answer for the frame as it was stored"
    assert any("KEY-FROZEN-MUTATED" in str(w.message) for w in seen), [str(w.message) for w in seen]


def test_a_frozen_numpy_result_restored_is_keyed_without_hashing_it(tmp_path):
    def build(cache_dir):
        c = Cash(cache_dir=str(cache_dir), register_magic=False)

        @c.cache(frozen=True)
        def make(n):
            return np.arange(n, dtype=np.float64)

        @c.cache
        def head(arr, k):
            return float(arr[:k].sum())

        return make, head

    make, head = build(tmp_path / ".cash")
    head(make(N), 5)
    _settle()
    make, head = build(tmp_path / ".cash")
    arr = make(N)
    with hashed_bytes() as hashed:
        assert head(arr, 5) == 10.0
    assert head.cache_info()["hits"] == 1
    assert hashed.bytes < arr.nbytes / 20, f"hashed {hashed.bytes} bytes of a {arr.nbytes}-byte array"
