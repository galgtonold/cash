"""A frozen function's polars frame is hashed once for its store.

``@cash.cache(frozen=True)`` hashes its result for the entry
(``content_digest``), and the RAM tier, keeping the frame as a clone, hashed
it again for the digest a hit checks the clone against: p08's cleaned log
frame (400 MB) twice, back to back. The tier now takes the entry's digest.
What a hit checks is unchanged: a frame over an array written after its
store is still a miss.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash import content_hashers
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

ROWS = 300_000  # over a MiB: kept as a clone


@pytest.fixture
def hashed(monkeypatch):
    calls = []
    real = content_hashers.hash_polars
    monkeypatch.setattr(content_hashers, "hash_polars", lambda f: calls.append(f) or real(f))
    return calls


def _metadata(frame):
    return {"execution_time": 1.0, "content_digest": content_hashers.builtin_hash(frame)}


def test_the_entry_s_digest_is_taken(hashed):
    df = pl.DataFrame({"x": np.arange(ROWS), "s": ["a", "bb", "ccc"] * (ROWS // 3)})
    metadata = _metadata(df)
    hashed.clear()
    backend = InMemoryBackend()
    backend.set("k", df, metadata)
    assert hashed == []
    _meta, hit = backend.get("k")
    assert hit.equals(df)


def test_without_one_the_clone_is_hashed(hashed):
    df = pl.DataFrame({"x": np.arange(ROWS)})
    InMemoryBackend().set("k", df, {"execution_time": 1.0})
    assert len(hashed) == 1


def test_a_frame_over_an_array_written_after_its_store_is_still_a_miss():
    a = np.arange(ROWS, dtype=np.float64)
    series = pl.Series("x", a)  # polars holds a's memory itself
    backend = InMemoryBackend()
    backend.set("k", series, _metadata(series))
    assert backend.get("k")[1][0] == 0.0
    a[0] = -1.0
    assert backend.get("k") == (None, None)
    assert memory_backend._CHECKED.known is None
