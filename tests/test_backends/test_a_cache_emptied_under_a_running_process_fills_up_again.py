"""A process's byte total must follow what others remove from the folder.

Each process counted the folder once, at its first write, and afterwards only
added its own writes and subtracted its own removals. After ``cash clear``,
another process's evictions or files deleted by hand, a long-running process
still believed the folder was full: once its stale total passed the cap,
every write started an eviction that deleted the process's own fresh results
from a nearly empty folder, could never reach its target, and warned
CACHE-THRASH with the advice to raise the cap.

10 MB cap and 0.5 MB values, so 16 entries are 8 MB and ten more fit an
emptied folder twice over.
"""

from __future__ import annotations

import os
import shutil
import warnings

import pytest

from cash.backends import FileBackend
from cash.backends.entry_format import ENTRY_SUFFIX
from cash.backends.file_eviction import FileEvictor

HALF_MB = 500_000
CAP = 10_000_000


def _backend(cache_dir):
    return FileBackend(str(cache_dir), max_size_bytes=CAP, flush_interval=0)


def _put(b, key):
    b.set(key, b"x" * HALF_MB, {"execution_time": 1.0})
    b._writes.wait_all()


def _held(b, keys):
    return [k for k in keys if os.path.exists(b._get_path(k))]


def _entries(cache_dir) -> list[str]:
    return [n for n in os.listdir(cache_dir) if n.endswith(ENTRY_SUFFIX)]


def _fill(b, count=16):
    old = [f"old-{i}" for i in range(count)]
    for key in old:
        _put(b, key)
    return old


def _write_new(b, count=10):
    new = [f"new-{i}" for i in range(count)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for key in new:
            _put(b, key)
    thrash = [str(w.message) for w in rec if getattr(w.message, "code", "") == "CACHE-THRASH"]
    return new, thrash


def test_results_stored_after_the_folder_was_removed_are_kept(tmp_path):
    """``cash clear --all`` under a running process: the folder goes whole."""
    cache = tmp_path / "c"
    b = _backend(cache)
    _fill(b)
    shutil.rmtree(cache)

    new, thrash = _write_new(b)

    assert _held(b, new) == new, "fresh results were evicted from a folder far under its cap"
    assert not thrash, thrash
    b.shutdown()


def test_results_stored_after_a_partial_clear_are_kept(tmp_path):
    """``cash clear --function``: entries go and the folder's stamp is renewed."""
    cache = tmp_path / "c"
    b = _backend(cache)
    for key in _fill(b):
        os.remove(b._get_path(key))
    b.stamp.bump()

    new, thrash = _write_new(b)

    assert _held(b, new) == new
    assert not thrash, thrash
    b.shutdown()


def test_entries_another_process_removed_are_noticed_at_the_next_count(tmp_path, monkeypatch):
    """Another process evicting, or files deleted by hand: nothing marks the
    folder, so the total is counted again once it is old enough."""
    monkeypatch.setattr(FileEvictor, "SIZE_RESCAN_INTERVAL", 0.0)
    cache = tmp_path / "c"
    b = _backend(cache)
    for key in _fill(b):
        os.remove(b._get_path(key))

    new, thrash = _write_new(b)

    assert _held(b, new) == new
    assert not thrash, thrash
    b.shutdown()


def test_entries_removed_by_hand_moments_ago_cost_no_fresh_result(tmp_path):
    """Inside the interval, with nothing marking the folder: the first pass
    over the cap ranks the folder, counts it on the same occasion, and finds
    nothing to evict."""
    cache = tmp_path / "c"
    b = _backend(cache)
    for key in _fill(b):
        os.remove(b._get_path(key))

    new, thrash = _write_new(b)

    assert _held(b, new) == new
    assert not thrash, thrash
    assert b.evictor.current_bytes == sum(os.path.getsize(cache / n) for n in _entries(cache))
    b.shutdown()


def test_ranked_entries_another_process_removed_cost_no_fresh_result(tmp_path):
    """The folder was ranked already (this process has evicted), so the next
    pass does not rank it again: it meets a ranked entry that is gone, and
    counts before choosing another victim."""
    cache = tmp_path / "c"
    b = _backend(cache)
    old = _fill(b, count=24)  # over the cap once: ranked, some evicted
    assert b.evictor.ranked and len(_entries(cache)) < 24, "precondition: a ranking exists"
    for key in _held(b, old):
        os.remove(b._get_path(key))

    new, thrash = _write_new(b)

    assert _held(b, new) == new
    assert not thrash, thrash
    b.shutdown()


def test_a_folder_that_is_really_full_is_still_kept_to_its_cap(tmp_path, monkeypatch):
    """The control: counting again must not stop eviction."""
    monkeypatch.setattr(FileEvictor, "SIZE_RESCAN_INTERVAL", 0.0)
    cache = tmp_path / "c"
    b = _backend(cache)
    _fill(b, count=30)
    on_disk = sum(os.path.getsize(cache / n) for n in _entries(cache))
    assert on_disk <= CAP
    assert len(_entries(cache)) < 30
    assert b.evictor.current_bytes == on_disk
    b.shutdown()


def test_a_full_folder_is_not_counted_on_every_eviction(tmp_path, monkeypatch):
    """Counting costs a ``stat`` per entry: once at the first write and once
    per ranking, which lists the folder anyway, not once per pass."""
    cache = tmp_path / "c"
    b = _backend(cache)
    counted, ranked, passes = [], [], []
    scan, rebuild, evict = b.evictor.scan_size_bytes, b.evictor.rebuild_queue, b.evictor.evict
    monkeypatch.setattr(b.evictor, "scan_size_bytes", lambda: counted.append(1) or scan())
    monkeypatch.setattr(b.evictor, "rebuild_queue", lambda: ranked.append(1) or rebuild())
    monkeypatch.setattr(b.evictor, "evict", lambda: passes.append(1) or evict())
    _fill(b, count=120)
    assert len(_entries(cache)) < 120, "precondition: the cap evicted"
    assert len(passes) == 120
    assert len(counted) <= 1 + len(ranked), (len(counted), len(ranked))
    assert len(counted) <= 12, f"the folder was counted {len(counted)} times in {len(passes)} writes"
    b.shutdown()


@pytest.mark.timeout(120)
def test_a_running_script_keeps_what_it_stores_after_cash_clear(tmp_path):
    """The whole path: a script fills 8 MB of a 10 MB cap, ``cash clear
    --all`` runs in another process, the script stores 5 MB more."""
    import textwrap

    from tests._scripts import run_python

    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import glob, os, subprocess, sys, warnings
            import cash
            from cash.backends import file_backend

            cache = os.environ["CASH_CACHE_DIR"]

            @cash.cache
            def f(n):
                return bytes([n % 251]) * 500_000

            def flush():
                for b in file_backend._LIVE_BACKENDS:
                    b._writes.wait_all()

            def on_disk():
                return sum(os.path.getsize(p) for p in glob.glob(cache + "/*.entry"))

            for n in range(16):
                f(n)
            flush()
            print("filled", on_disk())
            subprocess.run([sys.executable, "-m", "cash", "clear", "--all"], check=True, stdout=subprocess.DEVNULL)
            print("cleared", on_disk())
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                for n in range(100, 110):
                    f(n)
                flush()
            print("kept", on_disk())
            print("thrash", sum(getattr(w.message, "code", "") == "CACHE-THRASH" for w in rec))
            """
        ),
        encoding="utf-8",
    )
    out = dict(
        line.split()
        for line in run_python("job.py", cwd=tmp_path, env={"CASH_MAX_CACHE_SIZE": "10MB"}).stdout.splitlines()
    )
    assert int(out["filled"]) >= 8_000_000 and int(out["cleared"]) == 0, out
    assert int(out["kept"]) >= 5_000_000, f"fresh results were evicted from a nearly empty cache: {out}"
    assert out["thrash"] == "0"
