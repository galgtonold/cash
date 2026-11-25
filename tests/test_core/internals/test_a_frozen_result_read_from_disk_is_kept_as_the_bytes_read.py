"""A ``frozen=True`` result read from disk is kept by the RAM tier as the
bytes it was read out of: not walked for its size, not copied.

The first hit of a frozen entry in a process reads it from disk, hands
the value out and keeps it for every later hit (`frozen_handback`); the
RAM tier stores the entry beside it. For an object of a class of its own
that entry was the file's pickle stream already, but a frozenset or a
dict went through the RAM tier's ordinary store: sized by a walk of every
object in it and deep-copied, though the bytes just read are a copy of
their own. PyThaiNLP's ``get_corpus``, a frozenset of 60,000 words, paid
130-370 ms of ``read.fetch`` on its first hit in each process, against
2 ms when not declared frozen. Kept as the stream, it is sized by the
stream's length, and a hit that cannot hand the kept object back (it was
changed) unpickles a new copy out of it, as a disk hit does.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash
from cash.backends import memory_backend

# A frozenset cannot be marked as a frozen result passed on: said, and beside the point.
pytestmark = pytest.mark.filterwarnings("ignore:.*KEY-FROZEN-NO-EFFECT")


def corpus():
    return frozenset(f"word{i:06d}" for i in range(60_000))


def table():
    return {f"k{i}": [i, f"v{i}"] for i in range(20_000)}


@pytest.fixture
def walks(monkeypatch):
    """What the RAM tier sized by a walk or deep-copied, by type."""
    seen = []
    footprint = memory_backend.memory_footprint
    deep_copy = memory_backend.InMemoryBackend._safe_deep_copy

    def counted_footprint(value, *args, **kwargs):
        seen.append(("size", type(value).__name__))
        return footprint(value, *args, **kwargs)

    def counted_copy(self, value, *args, **kwargs):
        seen.append(("copy", type(value).__name__))
        return deep_copy(self, value, *args, **kwargs)

    monkeypatch.setattr(memory_backend, "memory_footprint", counted_footprint)
    monkeypatch.setattr(memory_backend.InMemoryBackend, "_safe_deep_copy", counted_copy)
    return seen


@pytest.mark.parametrize("make", [corpus, table], ids=["frozenset", "dict"])
def test_the_first_hit_from_disk_keeps_the_bytes_read(tmp_path, walks, make):
    cache_dir = str(tmp_path / ".cash")
    writer = Cash(cache_dir=cache_dir, register_magic=False)
    stored = writer.cache(frozen=True)(make)()
    writer.backend.list_entries()  # waits for the disk write

    # A second Cash over the same folder: its RAM tier is empty, as a new
    # process's is.
    reader = Cash(cache_dir=cache_dir, register_magic=False)
    produce = reader.cache(frozen=True)(make)
    ram = reader.backend.backends[0]
    walks.clear()
    first = produce()
    assert produce.cache_info()["hits"] == 1
    assert first == stored
    assert walks == [], f"the RAM tier walked or copied the value it read from disk: {walks}"
    ((key, (metadata, kept)),) = ram._store.items()
    assert type(kept) is memory_backend._Pickled, "not kept as the bytes read"
    assert metadata["size"] == kept.nbytes
    assert produce() is first, "the next hit did not hand back the kept object"


def test_a_changed_kept_result_is_read_again_out_of_the_bytes(tmp_path):
    cache_dir = str(tmp_path / ".cash")
    writer = Cash(cache_dir=cache_dir, register_magic=False)
    stored = writer.cache(frozen=True)(table)()
    writer.backend.list_entries()

    reader = Cash(cache_dir=cache_dir, register_magic=False)
    produce = reader.cache(frozen=True)(table)
    first = produce()
    if produce() is not first:
        pytest.skip("the kept object is proved unchanged by the C extension's watchers")
    first["k1"].append("changed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = produce()
    assert fresh is not first
    assert fresh == stored, "the copy read out of the kept bytes is not the stored value"
    assert produce.cache_info()["misses"] == 0
