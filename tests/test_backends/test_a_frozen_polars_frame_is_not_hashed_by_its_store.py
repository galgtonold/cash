"""A frozen function's polars frame is hashed once, for its entry.

``@cash.cache(frozen=True)`` hashes its result for the entry
(``content_digest``). The RAM tier kept the frame as a clone and hashed it
too, for the digest a hit checked the clone against: p08's cleaned log frame
(400 MB) twice, back to back. It took the entry's digest instead; now it
copies the frame (`memory_backend._store_polars`) and hashes nothing, and a
frame over an array written after its store is served as stored.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash import content_hashers
from cash.backends.memory_backend import InMemoryBackend

ROWS = 300_000  # over a MiB: was kept as a clone


@pytest.fixture
def hashed(monkeypatch):
    calls = []
    real = content_hashers.hash_polars
    monkeypatch.setattr(content_hashers, "hash_polars", lambda f: calls.append(f) or real(f))
    return calls


def _metadata(frame):
    return {"execution_time": 1.0, "content_digest": content_hashers.builtin_hash(frame)}


@pytest.mark.parametrize("with_digest", [True, False], ids=["the entry's digest", "none"])
def test_the_store_hashes_nothing(hashed, with_digest):
    df = pl.DataFrame({"x": np.arange(ROWS), "s": ["a", "bb", "ccc"] * (ROWS // 3)})
    metadata = _metadata(df) if with_digest else {"execution_time": 1.0}
    hashed.clear()
    backend = InMemoryBackend()
    backend.set("k", df, metadata)
    _meta, hit = backend.get("k")
    assert hashed == []
    assert hit.equals(df)


def test_a_frame_over_an_array_written_after_its_store_is_served_as_stored():
    a = np.arange(ROWS, dtype=np.float64)
    series = pl.Series("x", a)  # polars holds a's memory itself
    backend = InMemoryBackend()
    backend.set("k", series, _metadata(series))
    a[0] = -1.0
    assert backend.get("k")[1][0] == 0.0
