"""The disk cache's cap is said when caching starts, and its first eviction too.

The cap is sized to the machine unless set -- a quarter of the free space, 8 to
100 GiB -- and eviction at it is otherwise silent: a cache can grow to many GiB,
or lose entries, with nothing on screen to say why. So the cap is shown once
per process and cache folder, with where the number comes from, and the first
eviction says what it removed. Both are quiet notices (INFO records on
``cash.storage``), not warnings: nothing is wrong.

Free space is made up through the pure functions and a patched
``free_bytes_on_volume``, so every branch runs on any machine.
"""

from __future__ import annotations

import logging

import pytest

import cash.backends.adaptive_caps as caps
from cash import Cash
from cash.backends import FileBackend
from cash.backends.file_eviction import FileEvictor

GIB = 1024**3
MB = 1_000_000


def _notices():
    """The notices module, imported per test so that each one fails, rather
    than the file failing to collect, on a tree without it."""
    from cash.backends import budget_notices

    return budget_notices


# -- where the number comes from, in words -----------------------------------


@pytest.mark.parametrize(
    ("free", "cap", "says"),
    [
        (60 * GIB, 15 * GIB, "a quarter of the free disk space"),
        (20 * GIB, 8 * GIB, "the 8 GiB minimum; a quarter of the free disk space would be less"),
        (2048 * GIB, 100 * GIB, "the 100 GiB maximum; a quarter of the free disk space would be more"),
        (4 * GIB, int(0.8 * 4 * GIB), "80% of the free disk space, as that is less than the usual 8 GiB minimum"),
        (0, 8 * GIB, "the 8 GiB default, as the free disk space could not be measured"),
    ],
)
def test_the_reason_names_the_rule_that_decided(free, cap, says):
    assert caps.adaptive_disk_cap(free) == cap
    assert caps.disk_cap_reason(free) == says


def test_what_the_cache_holds_counts_as_free():
    """The cap counts the cache's own bytes as room (`adaptive_disk_cap_for`),
    so the reason must too: 20 GiB free plus 40 GiB held is a quarter of 60."""
    assert caps.disk_cap_reason(20 * GIB, 40 * GIB) == "a quarter of the free disk space"
    assert caps.disk_cap_reason(20 * GIB, 0).startswith("the 8 GiB minimum")


def test_the_line_names_folder_size_reason_and_the_setting():
    DiskBudget, describe_budget = _notices().DiskBudget, _notices().describe_budget
    adaptive = describe_budget(DiskBudget("/data/.cash", 26 * GIB, "a quarter of the free disk space"))
    assert adaptive == (
        "caching in /data/.cash, up to 26.0 GiB (a quarter of the free disk space; set max_cache_size to change it)"
    )
    # A cap the user set reads back as they wrote it, not in binary units.
    assert describe_budget(DiskBudget("/data/.cash", 2 * 10**9, None)) == (
        "caching in /data/.cash, up to 2 GB (set by max_cache_size)"
    )


# -- the file tier's own account of its cap -----------------------------------


@pytest.fixture
def volume(monkeypatch):
    """Free space and the cache's footprint, both ours to set."""
    state = {"free": 60 * GIB, "own": 0}
    monkeypatch.setattr(caps, "free_bytes_on_volume", lambda path: state["free"])
    monkeypatch.setattr(FileEvictor, "scan_size_bytes", lambda self: state["own"])
    return state


def test_an_adaptive_cap_is_reported_as_the_cap_it_enforces(tmp_path, volume):
    volume["own"] = 4 * GIB
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=caps.resolve_disk_cap(str(tmp_path)), adaptive_cap=True)
    budget = b.disk_budget()
    assert budget == _notices().DiskBudget(b.cache_dir, 16 * GIB, "a quarter of the free disk space")
    # The number the eviction then keeps to is the same one.
    b.evictor.ensure_size_scanned()
    assert b.evictor.max_size_bytes == budget.cap
    b.shutdown()


def test_a_set_cap_is_reported_as_set(tmp_path):
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=5 * MB)
    assert b.disk_budget() == _notices().DiskBudget(b.cache_dir, 5 * MB, None)
    b.shutdown()


def test_an_uncapped_tier_has_nothing_to_say(tmp_path):
    b = FileBackend(str(tmp_path / "c"))
    assert b.disk_budget() is None
    b.shutdown()


# -- the decorator: the first store logs the cap, once per folder -------------


def _storage_records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "cash.storage" and r.levelno == logging.INFO]


def test_the_first_store_logs_the_cap_once_per_folder(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="cash.storage")
    folder = tmp_path / "c"
    first = Cash(cache_dir=str(folder), max_cache_size="50MB", register_magic=False)

    @first.cache
    def f(x):
        return x * 2

    f(1)
    f(2)
    first.backend.shutdown()
    # A second Cash on the same folder, in the same process: already said.
    second = Cash(cache_dir=str(folder), max_cache_size="50MB", register_magic=False)

    @second.cache
    def g(x):
        return x * 3

    g(1)
    second.backend.shutdown()

    lines = _storage_records(caplog)
    assert lines == [f"caching in {first.backend.local_dir}, up to 50 MB (set by max_cache_size)"], lines


def test_a_log_nobody_reads_does_not_use_up_the_notice(tmp_path, caplog):
    """With ``cash.storage`` below INFO nothing is shown, so nothing is claimed:
    ``%cash_on`` in the same kernel must still show the cap."""
    caplog.set_level(logging.WARNING, logger="cash.storage")
    called = []
    _notices().announce_budget(str(tmp_path), lambda: called.append(1))
    assert called == [], "the cap was measured for a line nobody sees"
    assert _notices().claim_budget_notice(str(tmp_path)), "the folder was claimed without being shown"


# -- the first eviction is said, later ones are not ---------------------------


def _put(b, key, size, seconds=1.0):
    b.set(key, b"x" * size, {"execution_time": seconds})
    b._writes.wait_all()


def test_the_first_eviction_is_said_once(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="cash.storage")
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=5 * MB, flush_interval=0)
    for i in range(6):  # the sixth 1 MB entry takes the cache over 5 MB
        _put(b, f"k{i}", MB)
    notices = b.take_storage_notices()
    assert len(notices) == 1, notices
    text = notices[0]
    assert text.startswith(f"the cache in {b.cache_dir} reached its 5 MB cap, so cash removed ")
    assert "worth least per byte" in text and "max_cache_size" in text

    for i in range(6, 12):  # more evictions
        _put(b, f"k{i}", MB)
    assert b.take_storage_notices() == [], "a later eviction was reported again"
    info = [m for m in _storage_records(caplog) if "reached its" in m]
    assert info == [text], "the first eviction is one INFO record, the rest are debug"
    assert any(r.levelno == logging.DEBUG and "evicted" in r.getMessage() for r in caplog.records)
    b.shutdown()


def test_nothing_is_said_while_the_cache_is_under_its_cap(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="cash.storage")
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=50 * MB, flush_interval=0)
    for i in range(5):
        _put(b, f"k{i}", MB)
    assert b.take_storage_notices() == []
    assert not [m for m in _storage_records(caplog) if "reached its" in m]
    b.shutdown()
