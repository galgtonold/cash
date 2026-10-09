"""The stored-key record is written in the background, several stores at a time.

It used to be read, modified and rewritten on the calling thread for every
persisted store. Now a store only notes the key; one writer task writes each
function's file, taking whatever was noted meanwhile.
"""

from __future__ import annotations

import threading
import time

import pytest

import cash.decorator.stored_keys as stored_keys
from cash.decorator.stored_keys import StoredKeyRecord

pytestmark = pytest.mark.core


def _no_ledger(state):
    return None


def test_a_burst_of_stores_is_written_in_fewer_writes(tmp_path, monkeypatch):
    record = StoredKeyRecord(lambda: str(tmp_path))
    gate = threading.Event()
    writes = []
    real = stored_keys.replace_with_retry

    def slow_replace(tmp, path):
        gate.wait(5)
        writes.append(path)
        real(tmp, path)

    monkeypatch.setattr(stored_keys, "replace_with_retry", slow_replace)
    for i in range(20):
        record.note_stored("mod.f", f"mod.f:s:{i}:d", None, _no_ledger)
    gate.set()
    record.flush()
    assert 1 <= len(writes) < 20
    assert set(StoredKeyRecord(lambda: str(tmp_path)).read("mod.f")["keys"]) == {f"mod.f:s:{i}:d" for i in range(20)}
    record.close()


def test_a_note_is_read_back_before_it_is_written(tmp_path):
    record = StoredKeyRecord(lambda: str(tmp_path))
    record.note_ram_only("mod.f", "mod.f:s:a:d", "under the floor", _no_ledger)
    assert record.read("mod.f")["ram_only"]["mod.f:s:a:d"][1] == "under the floor"
    assert not (tmp_path / ".keys").exists(), "a RAM-only result was written on its own"


def test_close_writes_what_was_only_noted(tmp_path):
    record = StoredKeyRecord(lambda: str(tmp_path))
    record.note_ram_only("mod.f", "mod.f:s:a:d", "under the floor", _no_ledger)
    record.note_warning_shown("mod.f", "abc")
    record.close()
    doc = StoredKeyRecord(lambda: str(tmp_path)).read("mod.f")
    assert "mod.f:s:a:d" in doc["ram_only"]
    assert "abc" in doc["warned"]


def test_a_key_stored_after_it_was_kept_in_ram_is_only_a_stored_key(tmp_path):
    record = StoredKeyRecord(lambda: str(tmp_path))
    record.note_ram_only("mod.f", "mod.f:s:a:d", "under the floor", _no_ledger)
    record.note_stored("mod.f", "mod.f:s:a:d", 60, _no_ledger)
    record.close()
    doc = StoredKeyRecord(lambda: str(tmp_path)).read("mod.f")
    assert list(doc["keys"]) == ["mod.f:s:a:d"]
    assert doc["ram_only"] == {}


def test_no_local_directory_means_no_record(tmp_path):
    record = StoredKeyRecord(lambda: None)
    record.note_stored("mod.f", "mod.f:s:a:d", None, _no_ledger)
    record.close()
    assert record.read("mod.f")["keys"] == {}


def _written(tmp_path, key) -> bool:
    return key in StoredKeyRecord(lambda: str(tmp_path)).read("mod.f")["keys"]


def test_a_record_is_written_at_most_once_per_interval(tmp_path, monkeypatch):
    """A miss of a cheap function rewrote the whole record each time: a read,
    a JSON dump and a rename per miss, more than the call itself."""
    record = StoredKeyRecord(lambda: str(tmp_path))
    writes = []
    real = stored_keys.replace_with_retry

    def counting_replace(tmp, path):
        writes.append(path)
        real(tmp, path)

    monkeypatch.setattr(stored_keys, "replace_with_retry", counting_replace)
    for i in range(10):
        record.note_stored("mod.f", f"mod.f:s:{i}:d", None, _no_ledger)
        record._writes.wait_all()  # each store lands after the last write ended
    assert len(writes) == 1  # the first; the rest are held for the interval
    assert set(record.read("mod.f")["keys"]) == {f"mod.f:s:{i}:d" for i in range(10)}  # answered from memory
    record.close()
    assert len(writes) == 2
    assert all(_written(tmp_path, f"mod.f:s:{i}:d") for i in range(10))


def test_a_held_write_is_made_once_the_interval_has_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(StoredKeyRecord, "WRITE_INTERVAL", 0.2)
    record = StoredKeyRecord(lambda: str(tmp_path))
    record.note_stored("mod.f", "mod.f:s:0:d", None, _no_ledger)
    record._writes.wait_all()
    record.note_stored("mod.f", "mod.f:s:1:d", None, _no_ledger)
    deadline = time.monotonic() + 10
    while not _written(tmp_path, "mod.f:s:1:d") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _written(tmp_path, "mod.f:s:1:d"), "a held write never happened without a flush"
    record.close()


def test_a_miss_reason_reads_the_record_without_a_stat_per_miss(tmp_path, monkeypatch):
    record = StoredKeyRecord(lambda: str(tmp_path))
    record.note_stored("mod.f", "mod.f:s:0:d", None, _no_ledger)
    record.flush()
    stats = []
    real = stored_keys.os.stat

    def counting_stat(path, *args, **kwargs):
        stats.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(stored_keys.os, "stat", counting_stat)
    for _ in range(50):
        assert "mod.f:s:0:d" in record.read("mod.f")["keys"]
    assert len(stats) <= 1
    record.close()


def test_another_process_s_write_shows_once_the_interval_has_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(StoredKeyRecord, "WRITE_INTERVAL", 0.2)
    reader = StoredKeyRecord(lambda: str(tmp_path))
    writer = StoredKeyRecord(lambda: str(tmp_path))
    writer.note_stored("mod.f", "mod.f:s:0:d", None, _no_ledger)
    writer.flush()
    assert "mod.f:s:0:d" in reader.read("mod.f")["keys"]
    time.sleep(0.05)  # a new mtime for the next write
    writer.note_stored("mod.f", "mod.f:s:1:d", None, _no_ledger)
    writer.flush()
    time.sleep(0.25)
    assert "mod.f:s:1:d" in reader.read("mod.f")["keys"]
    writer.close()
    reader.close()


def test_a_miss_is_answered_from_an_index_as_from_the_whole_record(tmp_path, monkeypatch):
    """`miss_facts` answers from a view kept up to date as keys are noted;
    every answer must be the one the whole record (`read`) gives."""
    import json
    import random

    from cash.decorator.call_state import PROCESS_STARTED
    from cash.decorator.stored_keys import facts_from_doc

    rng = random.Random(7)
    record = StoredKeyRecord(lambda: str(tmp_path))
    path = record.path("mod.f")
    (tmp_path / ".keys").mkdir()
    earlier = {f"mod.f:old{i % 3}:d:a{i}": [PROCESS_STARTED - 100 + i, None] for i in range(30)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"func": "mod.f", "keys": earlier, "ram_only": {}, "states": {}, "warned": {}}, fh)
    monkeypatch.setattr(StoredKeyRecord, "KEYS_MAX", 24)  # trimming happens within the run
    states = ["s1", "s2", "old0", "old1"]
    for step in range(400):
        key = f"mod.f:{rng.choice(states)}:{rng.choice(['d', 'e'])}:a{rng.randrange(40)}"
        if rng.random() < 0.7:
            record.note_stored("mod.f", key, None, _no_ledger)
        else:
            record.note_ram_only("mod.f", key, "under the floor", _no_ledger)
        if rng.random() < 0.05:
            record.flush()
        query = f"mod.f:{rng.choice(states + ['s3'])}:{rng.choice(['d', 'e'])}:a{rng.randrange(45)}"
        assert record.miss_facts("mod.f", query) == facts_from_doc(record.read("mod.f"), query), step
    record.close()


def test_a_miss_reads_no_whole_record(tmp_path, monkeypatch):
    monkeypatch.setattr(StoredKeyRecord, "WRITE_INTERVAL", 3600.0)  # no write lands meanwhile
    record = StoredKeyRecord(lambda: str(tmp_path))
    for i in range(60):
        record.note_stored("mod.f", f"mod.f:s:d:a{i}", None, _no_ledger)
    record.flush()
    applied = []
    real = stored_keys._Changes.apply
    monkeypatch.setattr(stored_keys._Changes, "apply", lambda self, doc: (applied.append(1), real(self, doc)))
    record.miss_facts("mod.f", "mod.f:s:d:x")  # builds the view once
    for i in range(100):
        record.note_stored("mod.f", f"mod.f:s:d:b{i % 30}", None, _no_ledger)
        record.miss_facts("mod.f", f"mod.f:s:d:c{i}")
    assert not applied, "a miss merged the whole record again"
    record.close()
