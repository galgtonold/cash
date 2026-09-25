"""What the disk cap evicts is noted, so a later miss can say so.

A miss on an entry the cap removed read "evicted or cleared" at best, and a
recompute of minutes said nothing. Each eviction now appends a note per entry
(``_evicted.log``: the entry's file name, a SHA-256 of its key, when, its
compute seconds and size -- never the key or the value). A miss looks its key
up there: ``explain()`` and the per-call line say "evicted to make room", and a
recompute over the 2 s "worth telling" floor warns ``CACHE-EVICTED-RECOMPUTE``
once per function. A store forgets the note; clearing the cache drops them all.
"""

from __future__ import annotations

import os
import time
import warnings

from cash import Cash
from cash.backends import FileBackend
from cash.backends._writes import all_pending_writes
from cash.backends.cache_dir import is_cash_file

MB = 1_000_000


def _log_module():
    """Imported per test, so each fails rather than the file failing to
    collect on a tree without it."""
    from cash.backends import eviction_log

    return eviction_log


def _put(b, key, size=MB, seconds=1.0):
    b.set(key, b"x" * size, {"execution_time": seconds})
    b._writes.wait_all()


def _evicted_keys(b, keys):
    return [k for k in keys if not os.path.exists(b._get_path(k))]


# -- the log itself -----------------------------------------------------------


def test_notes_survive_the_process_and_hold_no_key(tmp_path):
    el = _log_module()
    stem = "ab" * 32
    el.EvictionLog(str(tmp_path), _untracked).record([(stem, el.EvictionNote(1_700_000_000.0, 12.5, 4096))])

    fresh = el.EvictionLog(str(tmp_path), _untracked)  # another process
    assert fresh.lookup(stem) == el.EvictionNote(1_700_000_000.0, 12.5, 4096)
    assert fresh.lookup("cd" * 32) is None


def test_a_store_forgets_the_note_and_a_clear_drops_them_all(tmp_path):
    el = _log_module()
    log = el.EvictionLog(str(tmp_path), _untracked)
    log.record([("a" * 64, el.EvictionNote(1.0, 3.0, 10)), ("b" * 64, el.EvictionNote(1.0, 3.0, 10))])
    assert log.lookup("a" * 64) is not None
    log.forget("a" * 64)
    assert el.EvictionLog(str(tmp_path), _untracked).lookup("a" * 64) is None, "the removal was not written"
    assert el.EvictionLog(str(tmp_path), _untracked).lookup("b" * 64) is not None

    log.remove()
    assert not os.path.exists(log.path)
    assert log.lookup("b" * 64) is None


def test_the_file_is_read_once_per_process(tmp_path, monkeypatch):
    el = _log_module()
    log = el.EvictionLog(str(tmp_path), _untracked)
    log.record([("a" * 64, el.EvictionNote(1.0, 3.0, 10))])
    reads = []
    real = el.EvictionLog._read
    monkeypatch.setattr(el.EvictionLog, "_read", lambda self: reads.append(1) or real(self))
    for _ in range(50):
        log.lookup("a" * 64)
        log.lookup("f" * 64)
    assert reads == [1]


def test_past_its_size_it_keeps_the_newest_notes(tmp_path, monkeypatch):
    el = _log_module()
    monkeypatch.setattr(el.EvictionLog, "MAX_NOTES", 10)
    monkeypatch.setattr(el.EvictionLog, "COMPACT_AT_BYTES", 10 * 128)
    log = el.EvictionLog(str(tmp_path), _untracked)
    for i in range(40):
        log.record([(f"{i:064x}", el.EvictionNote(float(i), 1.0, 100))])

    assert os.path.getsize(log.path) <= 10 * 128
    fresh = el.EvictionLog(str(tmp_path), _untracked)
    kept = [i for i in range(40) if fresh.lookup(f"{i:064x}") is not None]
    assert kept == list(range(30, 40)), kept


def test_the_log_is_one_of_cashs_own_files():
    """``cash clear`` removes only what cash wrote; it must know this file."""
    from cash.backends import cache_dir

    assert cache_dir.EVICTIONS_FILENAME == "_evicted.log"
    assert is_cash_file(cache_dir.EVICTIONS_FILENAME)


# -- the file tier notes what its cap evicts ----------------------------------


def test_the_cap_notes_each_entry_it_evicts(tmp_path):
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=5 * MB, flush_interval=0)
    keys = [f"k{i}" for i in range(8)]
    for k in keys:
        _put(b, k, seconds=3.0)
    gone = _evicted_keys(b, keys)
    assert gone, "nothing was evicted, so nothing is under test"
    for k in gone:
        note = b.eviction_note(k)
        assert note is not None and note.seconds == 3.0 and note.size > MB, (k, note)
    kept = [k for k in keys if k not in gone]
    assert all(b.eviction_note(k) is None for k in kept)
    with open(os.path.join(b.cache_dir, "_evicted.log"), encoding="ascii") as fh:
        body = fh.read()
    assert "k0" not in body.split(), "a key reached the log"

    # Stored again: no longer evicted.
    _put(b, gone[0], seconds=3.0)
    assert b.eviction_note(gone[0]) is None

    b.clear()
    assert b.eviction_note(gone[1]) is None
    assert not os.path.exists(os.path.join(b.cache_dir, "_evicted.log"))
    b.shutdown()


def test_a_deleted_entry_is_not_noted_as_evicted(tmp_path):
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=50 * MB, flush_interval=0)
    _put(b, "k")
    b.delete("k")
    assert b.eviction_note("k") is None
    b.shutdown()


# -- the decorator ------------------------------------------------------------


def _big(i):
    time.sleep(0.02)
    return bytes([i]) * 400_000


def _drain():
    for queue in all_pending_writes():
        queue.wait_all()


def _filled(folder):
    """A run that filled a 2 MB disk tier, then a new process's view of it:
    a second Cash over the folder, whose RAM tier holds nothing."""
    first = Cash(cache_dir=str(folder), max_cache_size="2MB", register_magic=False)
    fill = first.cache(_big)
    for i in range(8):
        fill(i)
    _drain()
    first.backend.shutdown()
    second = Cash(cache_dir=str(folder), max_cache_size="2MB", register_magic=False)
    return second, second.cache(_big)


def _evicted_args(f):
    return [i for i in range(8) if "evicted to make room" in str(f.explain(i))]


def test_explain_and_the_miss_say_evicted_to_make_room(tmp_path):
    c, f = _filled(tmp_path / "c")
    evicted = _evicted_args(f)
    assert evicted, "nothing was evicted, so nothing is under test"
    # The note is what licenses the wording: an entry deleted by hand keeps
    # the old, honest "evicted or cleared".
    kept = next(i for i in range(8) if i not in evicted)
    c.backend.delete(f.explain(kept).cache_key)
    by_hand = str(f.explain(kept))
    assert "evicted or cleared" in by_hand and "to make room" not in by_hand, by_hand
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f(evicted[0])
    entry = f.cache_info()
    assert entry["misses"] == 1
    assert not [w for w in caught if getattr(w.message, "code", "") == "CACHE-EVICTED-RECOMPUTE"], (
        "a 20 ms recompute is under the 2 s floor and must not warn"
    )


def test_an_expensive_recompute_of_an_evicted_result_warns_once(tmp_path, monkeypatch):
    from cash.backends import budget_notices

    _c, f = _filled(tmp_path / "c")
    evicted = _evicted_args(f)
    assert len(evicted) >= 2, evicted
    # The 20 ms body stands in for a 2 s one: the floor is the unit here.
    monkeypatch.setattr(budget_notices, "EVICTED_RECOMPUTE_WARN_SECONDS", 0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in evicted:
            f(i)
    ours = [w for w in caught if getattr(w.message, "code", "") == "CACHE-EVICTED-RECOMPUTE"]
    assert len(ours) == 1, [str(w.message) for w in caught]
    text = str(ours[0].message)
    assert "had been evicted from the disk cache to make room" in text
    assert "its 2 MB cap" in text and "raise max_cache_size above 2 MB" in text
    assert f.cache_info()["warnings"][-1]["code"] == "CACHE-EVICTED-RECOMPUTE"


def _untracked():
    import contextlib

    return contextlib.nullcontext()
