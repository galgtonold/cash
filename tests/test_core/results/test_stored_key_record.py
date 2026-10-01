"""The stored-key record is written in the background, several stores at a time.

It used to be read, modified and rewritten on the calling thread for every
persisted store. Now a store only notes the key; one writer task writes each
function's file, taking whatever was noted meanwhile.
"""

from __future__ import annotations

import threading

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
