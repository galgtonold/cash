"""A statement's superseded versions are pruned by what they are worth.

One notebook held nine ~700 MB versions of one 1.4 s feature build, another
five to seven of each ~500 MB cleaning frame -- 10 GB for 200 MB of input,
none of it evicted, because the byte cap is a quarter of the free disk. The
integration arm is
``tests/test_notebook_integration/storage/test_superseded_versions_are_pruned.py``.
"""

from __future__ import annotations

import os

from cash.backends.file_backend import FileBackend
from cash.backends.versions import (
    BYTES_PER_COMPUTE_SECOND,
    MAX_SUPERSEDED,
    Version,
    VersionIndex,
    superseded_to_drop,
)

MB = 1024 * 1024


def _versions(n, size, cost):
    return {f"k{i}": Version(f"k{i}", size, cost, used=float(i)) for i in range(n)}


# -- the policy ---------------------------------------------------------------


def test_a_big_cheap_value_keeps_only_the_newest_superseded_version():
    # A feature frame: 700 MB for 1.4 s of compute, nine versions
    versions = _versions(9, 700 * MB, 1.4)
    assert sorted(superseded_to_drop(versions, "k8")) == [f"k{i}" for i in range(7)]


def test_a_costly_small_result_keeps_many_versions():
    versions = _versions(30, 200, 120.0)
    drop = superseded_to_drop(versions, "k29")
    assert len(versions) - 1 - len(drop) == MAX_SUPERSEDED
    # the oldest go
    assert sorted(drop, key=lambda k: int(k[1:])) == [f"k{i}" for i in range(29 - MAX_SUPERSEDED)]


def test_what_a_version_affords_scales_with_its_compute():
    size = 100 * MB
    cost = 3 * size / BYTES_PER_COMPUTE_SECOND  # affords three superseded
    versions = _versions(6, size, cost)
    assert sorted(superseded_to_drop(versions, "k5")) == ["k0", "k1"]


def test_a_version_in_use_is_never_pruned():
    versions = _versions(5, 700 * MB, 0.1)
    assert sorted(superseded_to_drop(versions, "k4", in_use={"k0"})) == ["k1", "k2"]


def test_the_newest_superseded_version_stays_however_cheap():
    versions = _versions(2, 900 * MB, 0.0)
    assert superseded_to_drop(versions, "k1") == []


def test_recency_decides_which_is_newest_not_the_write_order():
    versions = _versions(3, 700 * MB, 1.0)
    versions["k0"].used = 99.0  # read after the others were written
    assert superseded_to_drop(versions, "k2") == ["k1"]


# -- the index ------------------------------------------------------------------


def _untracked():
    import contextlib

    return contextlib.nullcontext()


def test_the_index_is_read_back_by_another_process(tmp_path):
    a = VersionIndex(str(tmp_path), _untracked)
    a.record("s", "k0", 10, 1.0, 1.0)
    a.record("s", "k1", 10, 1.0, 2.0)
    a.forget(["k0"])
    a.touch("k1", 5.0)

    b = VersionIndex(str(tmp_path), _untracked)
    versions = b.record("s", "k2", 10, 1.0, 3.0)
    assert set(versions) == {"k1", "k2"}
    assert versions["k1"].used == 5.0


def test_a_torn_or_foreign_line_is_skipped(tmp_path):
    (tmp_path / "_versions.log").write_text("s k0 10 1.0\ngarbage\ns k1 10 1.0 2.0\n- \n", encoding="utf-8")
    versions = VersionIndex(str(tmp_path), _untracked).record("s", "k2", 10, 1.0, 3.0)
    assert set(versions) == {"k1", "k2"}


# -- the disk tier ----------------------------------------------------------------


def _set(backend, key, slot, size=2 * MB, cost=0.05):
    meta = {"execution_time": cost}
    if slot:
        meta["version_slot"] = slot
    backend.set(key, b"x" * size, meta)
    backend._writes.wait_all()


def _on_disk(backend, key):
    return os.path.exists(backend._get_path(key))


def test_writing_new_versions_removes_the_ones_not_worth_their_bytes(tmp_path):
    b = FileBackend(str(tmp_path), flush_interval=0)
    for i in range(5):
        _set(b, f"v{i}", "slot")
    assert [_on_disk(b, f"v{i}") for i in range(5)] == [False, False, False, True, True]
    b.shutdown()


def test_entries_without_a_slot_or_in_another_slot_are_left_alone(tmp_path):
    b = FileBackend(str(tmp_path), flush_interval=0)
    _set(b, "call-a", None)
    _set(b, "other", "slot-2")
    for i in range(4):
        _set(b, f"v{i}", "slot")
    assert _on_disk(b, "call-a") and _on_disk(b, "other")
    b.shutdown()


def test_a_version_read_in_this_process_is_kept(tmp_path):
    b = FileBackend(str(tmp_path), flush_interval=0)
    _set(b, "v0", "slot")
    assert b.get("v0") is not None
    for i in range(1, 5):
        _set(b, f"v{i}", "slot")
    assert _on_disk(b, "v0")
    assert not _on_disk(b, "v1")
    b.shutdown()


def test_pruning_continues_across_processes(tmp_path):
    first = FileBackend(str(tmp_path), flush_interval=0)
    _set(first, "v0", "slot")
    _set(first, "v1", "slot")
    first.shutdown()

    second = FileBackend(str(tmp_path), flush_interval=0)
    _set(second, "v2", "slot")
    assert not _on_disk(second, "v0")
    assert _on_disk(second, "v1") and _on_disk(second, "v2")
    second.shutdown()


def test_clearing_the_cache_forgets_the_versions(tmp_path):
    b = FileBackend(str(tmp_path), flush_interval=0)
    _set(b, "v0", "slot")
    b.clear()
    assert not (tmp_path / "_versions.log").exists()
    b.shutdown()
