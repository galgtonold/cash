"""The cap a user is shown must be the cap the cache is keeping to.

A user, after two days on one notebook::

    Holds:      908 entries, 21.19 GiB
    Max size:   auto -- disk 12.0 GiB, RAM 4.0 GiB

and the next morning, same directory, same machine::

    917 entries, 16.44 GiB   (cap: auto -- disk 13.2 GiB)

Filed as "the cap is not being honoured on the only thing that can honour it".
Eviction was in fact working -- the overnight drop from 21.19 to 16.44 GiB IS
eviction. Two other things were wrong, and between them they account for both
readings exactly:

1. ``cash info`` asked ``resolve_disk_cap`` = f(free), while the backend
   enforces ``adaptive_disk_cap_for`` = f(free + own). The printed number is
   therefore lower than the enforced one by ``0.25 * own`` -- 5.3 GiB at the
   21.19 GiB this cache held -- so the user is always shown a cap the cache
   looks to have blown. ``cash info`` walks the directory one line earlier to
   print ``Holds``, so it has ``own`` in hand and discards it.

2. The backend derives its cap ONCE, on the first write of the process, and
   never again. That user's kernel opened when the machine had ~118 GB free
   (0.25 * 118 GiB = 29.5 GiB) and was still enforcing that number two days
   later with 48 GB free. 21.19 GiB was under ITS cap and over every cap the
   machine would have derived that afternoon. The restart re-derived
   0.25 * (48 + 21.19) = 17.3 GiB and evicted down to 16.44 -- which freed
   4.75 GiB, so the next ``cash info`` read 52.75 GiB free and printed 13.2.

So: report the number being enforced, and enforce a number sized from the
volume as it is now, not as it was when the kernel started.
"""

from __future__ import annotations

import pytest

import cash.backends.adaptive_caps as caps
from cash.backends import FileBackend
from cash.backends.adaptive_caps import adaptive_disk_cap, adaptive_disk_cap_for

GIB = 1024**3


@pytest.fixture
def volume(tmp_path, monkeypatch):
    """A volume whose free space is ours to move, and a declared footprint.

    Same shape as ``test_adaptive_cap_stability``'s fixture and for the same
    reason: really writing tens of GiB is not sparse on NTFS and killed the
    Windows CI job, and anything under a 32 GiB volume pins the cap to
    ``DISK_FLOOR`` and measures nothing.

    Yields ``(cache, state)``; set ``state["free"]`` and ``state["own"]``.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    state = {"free": 118 * GIB, "own": 0}

    monkeypatch.setattr(caps, "free_bytes_on_volume", lambda path: state["free"])
    monkeypatch.setattr(FileBackend, "_scan_size_bytes", lambda self: state["own"])
    return cache, state


def _backend(cache, adaptive=True, cap=None):
    return FileBackend(
        str(cache),
        max_size_bytes=cap if cap is not None else caps.resolve_disk_cap(str(cache)),
        adaptive_cap=adaptive,
        flush_interval=0,
    )


def test_a_long_lived_kernel_does_not_keep_a_two_day_old_cap(volume):
    """A long-lived kernel, in miniature: the disk fills under a live backend."""
    cache, state = volume

    b = _backend(cache)
    b._ensure_size_scanned()
    opened_with = b._max_size_bytes
    assert opened_with == adaptive_disk_cap(118 * GIB), "fixture did not take"

    # Two days pass; four other sessions take the volume down to 48 GB free,
    # and this cache is holding 21.19 GiB of that.
    state["free"] = 48 * GIB
    state["own"] = int(21.19 * GIB)
    b._current_size_bytes = state["own"]
    b._refresh_adaptive_cap(force=True)
    now = b._max_size_bytes
    b.shutdown()

    assert now == adaptive_disk_cap_for(str(cache), state["own"]), (
        f"a live backend is still enforcing {now / GIB:.1f} GiB, derived from "
        f"the free space it saw when it opened; the volume now supports "
        f"{adaptive_disk_cap_for(str(cache), state['own']) / GIB:.1f} GiB"
    )
    assert now < opened_with


def test_a_write_re_derives_the_cap_when_the_volume_has_moved(volume):
    """Not a method a caller must remember: the write path does it."""
    cache, state = volume

    b = _backend(cache)
    b.set("k", {"variables": {"v": b"x" * 1024}}, {"execution_time": 1.0, "size": 1024, "key": "k"})
    b._writes.wait_all()
    assert b._max_size_bytes == adaptive_disk_cap(118 * GIB)

    state["free"] = 48 * GIB
    b._cap_derived_at = 0.0  # the throttle, not the behaviour, under test
    b.set("k2", {"variables": {"v": b"x" * 1024}}, {"execution_time": 1.0, "size": 1024, "key": "k2"})
    b._writes.wait_all()
    derived = b._max_size_bytes
    b.shutdown()

    assert derived == adaptive_disk_cap_for(str(cache), b._current_size_bytes), (
        "writing did not re-derive the cap after the volume moved"
    )


def test_an_explicit_cap_is_still_never_re_derived(volume):
    """The control. Re-deriving must not start overriding the user's number."""
    cache, state = volume
    chosen = 3 * GIB

    b = _backend(cache, adaptive=False, cap=chosen)
    b._ensure_size_scanned()
    state["free"] = 4 * GIB
    b._refresh_adaptive_cap(force=True)
    b.set("k", {"variables": {"v": b"x" * 1024}}, {"execution_time": 1.0, "size": 1024, "key": "k"})
    b._writes.wait_all()
    assert b._max_size_bytes == chosen
    b.shutdown()


def test_the_re_derivation_is_throttled(volume):
    """One `disk_usage` per write on a synced volume is not free.

    The walk that produces the footprint is already latched; this must not
    quietly reintroduce a per-write syscall.
    """
    cache, state = volume
    calls = {"n": 0}
    real = caps.free_bytes_on_volume

    def counted(path):
        calls["n"] += 1
        return real(path)

    caps.free_bytes_on_volume = counted
    try:
        b = _backend(cache)
        for i in range(25):
            b.set(f"k{i}", {"variables": {"v": b"x" * 1024}}, {"execution_time": 1.0, "size": 1024, "key": f"k{i}"})
        b._writes.wait_all()
        b.shutdown()
    finally:
        caps.free_bytes_on_volume = real

    assert calls["n"] <= 3, (
        f"25 writes measured free space {calls['n']} times; the re-derivation is meant to be throttled, not per-write"
    )


def test_cash_info_prints_the_cap_the_backend_would_enforce(volume, capsys, monkeypatch):
    """The number on screen is the number in force -- the user's actual complaint.

    The exact readings: 21.19 GiB held on a volume with 48 GB free. The
    old answer was 12.0 GiB (f(free)) and looked like a cap already blown by
    77%; the enforced one is 17.3 GiB (f(free + own)) and is not blown at all.
    """
    from types import SimpleNamespace

    from cash import __main__ as cli
    from cash.backends.adaptive_caps import human_bytes
    from cash.config import get_config

    cache, state = volume
    state["own"] = int(21.19 * GIB)
    state["free"] = 48 * GIB

    # `cash info` counts top-level *.entry itself. 21 GiB of real files is not
    # sparse on NTFS, so the count is stubbed -- it has its own tests, and
    # here it is only the courier for a footprint.
    monkeypatch.setattr(cli, "_entry_totals", lambda d: (908, state["own"]), raising=True)
    config = get_config()
    monkeypatch.setattr(config, "cache_dir", str(cache))
    monkeypatch.setattr(config, "max_cache_size", None)
    monkeypatch.setattr("cash.__main__.get_config", lambda **_: config)
    cli.cmd_info(SimpleNamespace())

    printed = capsys.readouterr().out
    line = next(l for l in printed.splitlines() if l.strip().startswith("Max size:"))
    enforced = adaptive_disk_cap_for(str(cache), state["own"])
    assert human_bytes(enforced) in line, (
        f"`cash info` says {line.strip()!r}; the backend would enforce {human_bytes(enforced)}"
    )
