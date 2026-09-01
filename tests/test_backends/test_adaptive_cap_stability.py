"""An adaptive cap must not shrink because the cache filled.

``resolve_disk_cap`` sizes the disk tier from FREE space, and free space
excludes whatever the cache has already written. So the cap fell as the cache
grew, the cache was then over a cap its own contents had caused, eviction
overshot to 90%, that freed space, the next process read a larger free figure
and raised the cap again. Simulated over sessions on a volume with 48 GiB free
when empty, it settles into a two-cycle -- roughly 13% of the cache discarded
every other session, evicted for no reason the workload asked for.

Reported by a user as ``CashCacheIneffectiveWarning`` at a 9.6 GiB cap, which
is what a ~48 GiB volume converges to.

Adding the cache's own size back in removes the loop: as the cache grows by N
bytes, free drops by N, and the sum is unchanged.
"""
from __future__ import annotations

import os

import pytest

import cash.backends.adaptive_caps as caps
from cash.backends import FileBackend
from cash.backends.adaptive_caps import adaptive_disk_cap, adaptive_disk_cap_for

GIB = 1024 ** 3


@pytest.fixture
def volume(tmp_path, monkeypatch):
    """A fake volume whose free space shrinks as the cache directory grows.

    That coupling is the whole subject: a probe returning a constant would
    make both the old and the new behaviour look identical.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    total_free_when_empty = 48 * GIB

    def fake_free(path):
        used = sum(f.stat().st_size for f in os.scandir(cache) if f.is_file())
        return total_free_when_empty - used

    monkeypatch.setattr(caps, "_free_bytes_on_volume", fake_free)
    return cache, total_free_when_empty


def _fill_to(cache, nbytes):
    """Occupy *nbytes* in the cache dir, sparsely (no real disk needed)."""
    for f in os.scandir(cache):
        if f.is_file():
            os.remove(f.path)
    with open(cache / "bulk.entry", "wb") as fh:
        fh.truncate(nbytes)


def test_the_cap_does_not_shrink_as_the_cache_fills(volume):
    """The defect, stated as the invariant it violates."""
    cache, free_when_empty = volume

    empty = FileBackend(str(cache), max_size_bytes=adaptive_disk_cap(free_when_empty),
                        adaptive_cap=True, flush_interval=0)
    empty._ensure_size_scanned()
    cap_when_empty = empty._max_size_bytes
    empty.shutdown()

    _fill_to(cache, cap_when_empty)          # the cache fills to that cap

    full = FileBackend(str(cache), max_size_bytes=adaptive_disk_cap(caps._free_bytes_on_volume(str(cache))),
                       adaptive_cap=True, flush_interval=0)
    full._ensure_size_scanned()
    cap_when_full = full._max_size_bytes
    full.shutdown()

    assert cap_when_full == cap_when_empty, (
        f"the cap moved from {cap_when_empty} to {cap_when_full} because the "
        f"cache filled; the space available TO the cache did not change"
    )


def test_an_explicit_cap_is_never_re_derived(volume):
    """The control. A user's number is theirs, whatever the volume says.

    Without this, 'always re-derive' would pass the arm above while quietly
    overriding `max_cache_size`.
    """
    cache, _ = volume
    chosen = 3 * GIB

    b = FileBackend(str(cache), max_size_bytes=chosen, adaptive_cap=False,
                    flush_interval=0)
    b._ensure_size_scanned()
    assert b._max_size_bytes == chosen
    b.shutdown()


def test_an_empty_cache_gets_the_same_cap_as_before(volume):
    """The correction must be invisible until there is something to correct for."""
    cache, free_when_empty = volume

    b = FileBackend(str(cache), max_size_bytes=None, adaptive_cap=True,
                    flush_interval=0)
    b._ensure_size_scanned()
    assert b._max_size_bytes == adaptive_disk_cap(free_when_empty)
    b.shutdown()


def test_the_cap_still_follows_the_volume(volume):
    """It must not become insensitive: less room really means a smaller cap.

    The fix removes the cache's own footprint from the signal, not the signal.
    """
    cache, free_when_empty = volume
    roomy = adaptive_disk_cap_for(str(cache), 0)

    # Something else on the volume eats 40 GiB.
    monkey = free_when_empty - 40 * GIB
    orig = caps._free_bytes_on_volume
    caps._free_bytes_on_volume = lambda path: monkey
    try:
        cramped = adaptive_disk_cap_for(str(cache), 0)
    finally:
        caps._free_bytes_on_volume = orig

    assert cramped < roomy, (
        f"cap did not fall when the volume filled: {roomy} -> {cramped}"
    )


def test_own_usage_is_added_to_free(tmp_path, monkeypatch):
    """The arithmetic, directly."""
    monkeypatch.setattr(caps, "_free_bytes_on_volume", lambda path: 30 * GIB)
    assert adaptive_disk_cap_for(str(tmp_path), 10 * GIB) == adaptive_disk_cap(40 * GIB)
    # A negative or nonsense own-size must not shrink the answer below free.
    assert adaptive_disk_cap_for(str(tmp_path), -5) == adaptive_disk_cap(30 * GIB)
