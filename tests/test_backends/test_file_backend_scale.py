"""Opening a big cache directory must not cost one file read per entry.

``_init_stats`` runs once per process on the first cache operation. It used to
``open()`` and unpickle every ``.meta`` in the directory to prefill
``_metadata_cache`` -- linear in the entry count, and it dominated startup:
98ms at 1k entries, 490ms at 5k, 2.1s at 20k, so roughly 10s at 100k and 100s
at 1M. Every process paid it, however few keys it went on to touch.

It was pure prefetch. ``get`` already reads the ``.meta`` from disk when a key
is not cached, so nothing depended on it for correctness -- and holding every
entry's metadata costs ~1.7KB each, 1.7GB at a million entries.

The size cap only ever needed the sizes, which ``scandir``/``stat`` gives 31x
faster. The LRU genuinely does need the metadata, so eviction loads it on
demand: the cost moves to the processes that actually evict.

The performance half of this is measured in ``benchmarks/bench_cache_scale.py``.
What is pinned here is the behaviour that makes it safe.
"""
from __future__ import annotations

import os
import pickle
import time

from cash.backends import FileBackend
from cash.backends.file_backend import CACHE_FORMAT_VERSION


def _seed(cache_dir, n, payload=b"x" * 256):
    """Write *n* entries directly, as a previous session would have left them.

    The ``CACHE_VERSION`` stamp is not optional: without it the format check
    treats the directory as written by an incompatible build and deletes every
    entry on first use, so the fixture would seed into a cache that promptly
    wiped itself.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "CACHE_VERSION").write_text(str(CACHE_FORMAT_VERSION),
                                             encoding="utf-8")
    paths = FileBackend(str(cache_dir))._get_paths
    keys = []
    for i in range(n):
        key = f"mod.f:state:{i}:args"
        keys.append(key)
        meta_path, data_path = paths(key)
        with open(meta_path, "wb") as fh:
            pickle.dump({"key": key, "size": len(payload),
                         "created_at": time.time(),
                         "last_access": time.time() + i,
                         "access_count": 1, "storage": ["DISK"]}, fh)
        with open(data_path, "wb") as fh:
            fh.write(payload)
    return keys


def test_opening_a_cache_does_not_read_every_entry(tmp_path):
    """The whole point: startup cost must not scale with the directory."""
    cache = tmp_path / "cache"
    _seed(cache, 25)

    backend = FileBackend(str(cache))
    backend.get("mod.f:state:0:args")          # forces _ensure_initialized

    assert len(backend._metadata_cache) <= 1, (
        f"init loaded {len(backend._metadata_cache)} metadata entries; it should "
        f"load none, and `get` should have cached only the key it was asked for"
    )
    assert backend._metadata_loaded is False


def test_the_size_total_is_still_right(tmp_path):
    """A cheaper scan is worthless if it stops counting correctly."""
    cache = tmp_path / "cache"
    _seed(cache, 20)
    on_disk = sum(f.stat().st_size for f in cache.iterdir()
                  if f.suffix in (".meta", ".data"))

    backend = FileBackend(str(cache))
    backend.get("mod.f:state:0:args")
    assert backend._current_size_bytes == on_disk


def test_a_key_is_still_readable_without_being_preloaded(tmp_path):
    """`get` reads the metadata itself -- that is why the prefetch was optional.

    Seeded through ``set`` rather than by writing files: this arm reads a
    VALUE back, so the payload has to go through the serializer the way a real
    session would write it.
    """
    cache = tmp_path / "cache"
    writer = FileBackend(str(cache))
    for i in range(10):
        writer.set(f"mod.f:state:{i}:args", {"payload": i})
    writer._writes.wait_all()

    reader = FileBackend(str(cache))          # fresh: nothing preloaded
    metadata, value = reader.get("mod.f:state:7:args")
    assert value == {"payload": 7}
    assert metadata is not None and metadata["key"] == "mod.f:state:7:args"
    assert len(reader._metadata_cache) == 1, "reading one key preloaded others"


def test_eviction_reaches_entries_this_process_never_touched(tmp_path):
    """The correctness risk of loading lazily, and the reason for the on-demand load.

    A fresh process ranking only the handful of keys it happens to have read
    would free almost nothing and let the directory grow past its cap forever.
    """
    cache = tmp_path / "cache"
    _seed(cache, 60, payload=b"x" * 5_000)
    before = len(list(cache.glob("*.data")))

    tight = FileBackend(str(cache), max_size_bytes=100_000)
    tight.set("newcomer", b"y" * 5_000,
              {"size": 5_000, "created_at": time.time(), "last_access": time.time()})
    tight._writes.wait_all()

    after = len(list(cache.glob("*.data")))
    assert after < before, (
        "eviction freed nothing: it can only see keys this process touched"
    )
    assert tight._metadata_loaded is True, "eviction did not load the metadata"
    assert tight._current_size_bytes <= 100_000 * 1.1


def test_deleting_an_untouched_key_updates_the_size(tmp_path):
    """Delete used to take the size from the metadata cache.

    With metadata loaded lazily an uncached key would have subtracted 0 and
    drifted the total upward forever. Measuring the files also counts the
    ``.meta`` bytes, which the cached ``size`` never included.
    """
    cache = tmp_path / "cache"
    _seed(cache, 5)

    backend = FileBackend(str(cache))
    backend.get("mod.f:state:0:args")          # cache exactly one key
    before = backend._current_size_bytes

    victim = "mod.f:state:3:args"              # never touched by this process
    meta_path, data_path = backend._get_paths(victim)
    on_disk = os.path.getsize(meta_path) + os.path.getsize(data_path)

    backend.delete(victim)
    assert backend._current_size_bytes == before - on_disk, (
        "the size total did not drop by what was actually removed"
    )
