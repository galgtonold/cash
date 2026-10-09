"""Whatever the RAM tier keeps as marshal bytes comes back as the value.

A big JSON-like variable in a notebook entry -- parsed records -- is kept
as bytes beside the rest of the entry, and a hit's copy reads it back out.
When that copy cannot be made, the tier handed back its stored entry
itself, bytes and all: ``with open(p) as f: records = [...]`` restored
``records`` as the tier's wrapper (the closed file a ``with`` leaves bound
is stored as a closed file again, which no later copy can copy), and the
next cell raised ``'_Marshalled' object is not iterable``. Every way out of
the tier -- a hit, the entry handed to the disk tier -- must hand back the
records, or nothing.
"""

from __future__ import annotations

import threading

from cash.backends import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook.closed_stream import stored_form

N = 5_000  # records: 10,000 containers, kept as bytes


def _records(n: int = N) -> list:
    return [{"id": i, "tags": ["a"]} for i in range(n)]


def _closed_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        pass
    return stored_form(f)  # as the notebook hands it to the backend


def _lock_maker() -> object:
    """Stored as a lock: copied once on the store, then never again."""

    class BecomesALock:
        def __reduce__(self):
            return threading.Lock, ()

    return BecomesALock()


def _plain(value, seen=None) -> bool:
    """Is *value* free of the tier's wrapper, wherever it is held?"""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return True
    seen.add(id(value))
    if type(value).__name__ == "_Marshalled":
        return False
    if type(value) is dict:
        return all(_plain(item, seen) for item in value.values())
    if type(value) in (list, tuple):
        return all(_plain(item, seen) for item in value)
    return True


def test_records_beside_a_closed_file_come_back_as_records(tmp_path):
    backend = InMemoryBackend()
    payload = {"variables": {"records": _records(), "f": _closed_file(tmp_path)}, "stdout": ""}
    backend.set("k", payload)
    for _ in range(2):
        variables = backend.get("k")[1]["variables"]
        assert type(variables["records"]) is list and variables["records"] == _records()
        assert variables["f"].closed


def test_records_beside_what_cannot_be_copied_again_come_back_as_records():
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"records": _records(), "lock": _lock_maker()}, "rng_state": (3, (1, 2))})
    got = backend.get("k")[1]
    assert _plain(got) and got["variables"]["records"] == _records()


def test_the_rest_of_such_an_entry_is_still_a_copy():
    """A hit that cannot copy the entry as a whole copies what it can of it:
    a list beside the lock is the caller's own, as on any other hit."""
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"records": _records(), "seen": [[1]], "lock": _lock_maker()}})
    first = backend.get("k")[1]["variables"]
    first["seen"][0].append("caller's")
    first["records"][0]["tags"].append("caller's")
    second = backend.get("k")[1]["variables"]
    assert second["seen"] == [[1]] and second["records"] == _records()


def test_two_names_for_the_records_still_share_them():
    backend = InMemoryBackend()
    records = _records()
    payload = {"variables": {"records": records, "same": records, "pair": (records, 1), "lock": _lock_maker()}}
    backend.set("k", payload)
    variables = backend.get("k")[1]["variables"]
    assert _plain(variables) and variables["records"] == _records()
    assert variables["same"] is variables["records"] and variables["pair"][0] is variables["records"]


def test_an_object_holding_the_records_that_cannot_be_copied_is_a_miss():
    """The records sit inside an object no copy can be made of: they cannot
    be read out of it, so the entry is not served at all."""

    class Box:
        def __init__(self, records):
            self.records = records
            self.lock = _lock_maker()

    backend = InMemoryBackend()
    records = _records()
    backend.set("k", {"variables": {"records": records, "box": Box(records)}})
    assert backend.peek_entry("k") is None
    assert backend.get("k") == (None, None)


def test_the_entry_as_peeked_holds_the_records(tmp_path):
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"records": _records(), "f": _closed_file(tmp_path)}})
    _metadata, value = backend.peek_entry("k")
    assert _plain(value) and value["variables"]["records"] == _records()


class _RecordingTier(InMemoryBackend):
    """A second tier that keeps what it was handed, as handed."""

    source_label = "DISK"
    cost_kind = "disk"

    def __init__(self) -> None:
        super().__init__()
        self.written: list = []

    def set(self, key, value, metadata=None, serializer=None):
        self.written.append(value)


def test_the_entry_written_on_from_ram_holds_the_records():
    ram, disk = InMemoryBackend(), _RecordingTier()
    tiered = TieredBackend([ram, disk])
    notebook = {"execution_time": 0.02, "cost_model_family": "_GENERIC", "cost_model_type_name": "list"}
    tiered.set("k", {"variables": {"records": _records(), "lock": _lock_maker()}}, notebook)
    assert disk.written == []

    assert tiered.persist_from_memory("k", rebuild_seconds=600.0) is True

    assert len(disk.written) == 1 and _plain(disk.written[0])
    assert disk.written[0]["variables"]["records"] == _records()
