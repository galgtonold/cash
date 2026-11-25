"""A chunked result whose chunks cannot be looked for is computed again.

An iterator's result is stored as a manifest and chunks, and the chunks
can go while the manifest stays (eviction reaches them one by one). So
before a hit is served, each chunk is looked for; one missing makes the
entry a miss, since a result with items left out is worse than a slow one.

When the look itself failed (a backend that raises), the entry was served
all the same, as if every chunk were there: the one case in which nothing
is known about them was treated as the best case.
"""

from __future__ import annotations

from cash import Cash
from cash.backends.file_backend import FileBackend


def _chunked(cash, runs):
    @cash.cache
    def numbers():
        runs.append(1)
        yield from range(6)

    return numbers


def _refusing_chunk_lookups(backend, monkeypatch) -> list:
    """Make *backend* raise when asked for a chunk's metadata."""
    asked = []
    real = backend.get_metadata

    def get_metadata(key, *args, **kwargs):
        if ":chunk_" in key:
            asked.append(key)
            raise OSError("the store cannot be reached")
        return real(key, *args, **kwargs)

    monkeypatch.setattr(backend, "get_metadata", get_metadata)
    return asked


def test_the_entry_is_not_served_when_its_chunks_cannot_be_looked_for(tmp_path, monkeypatch):
    backend = FileBackend(str(tmp_path / "store"), flush_interval=0)
    cash = Cash(backend=backend, register_magic=False)
    runs: list = []
    numbers = _chunked(cash, runs)
    assert list(numbers()) == list(range(6))
    assert list(numbers()) == list(range(6))
    assert len(runs) == 1, "the control: with its chunks in place the entry is served"

    asked = _refusing_chunk_lookups(backend, monkeypatch)
    assert list(numbers()) == list(range(6))
    assert asked, "the chunks were never looked for: this test checks nothing"
    assert len(runs) == 2, "an entry whose chunks could not be looked for was served"
