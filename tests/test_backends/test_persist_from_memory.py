"""``TieredBackend.persist_from_memory``: a RAM-only entry goes to disk when
rebuilding it would cost more than restoring it.

The notebook calls this at the end of a cell with a value's whole rebuild
cost -- its statement and the entries not on disk it came through -- where
``set`` could only see the statement's own compute time.
"""

from cash.backends import FileBackend, InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

# A notebook value: it carries a cost-model family, as the statement processor writes it.
NOTEBOOK = {
    "execution_time": 0.02,
    "cost_model_family": "_GENERIC",
    "cost_model_type_name": "int",
    "cost_model_size_bytes": 28,
}


def _tiered(tmp_path):
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "cache"))
    return TieredBackend([ram, disk]), ram, disk


def test_a_cheap_entry_stays_in_ram_when_set(tmp_path):
    """The control: what ``set`` decides by the statement's own time alone."""
    tiered, _ram, disk = _tiered(tmp_path)
    meta = dict(NOTEBOOK)
    tiered.set("k", 42, meta)
    assert meta["storage"] == ["RAM"] and meta["persist_skipped"] == "compute"
    assert disk.get("k") == (None, None)


def test_it_goes_to_disk_when_rebuilding_costs_more(tmp_path):
    tiered, ram, disk = _tiered(tmp_path)
    tiered.set("k", 42, dict(NOTEBOOK))

    assert tiered.persist_from_memory("k", rebuild_seconds=2.0) is True

    meta, value = disk.get("k")
    assert value == 42 and not meta.get("metadata_only")
    assert meta["rebuild_time"] == 2.0
    assert "DISK" in ram.peek_metadata("k")["storage"]
    assert "persist_skipped" not in ram.peek_metadata("k")


def test_it_stays_in_ram_when_rebuilding_is_cheap(tmp_path):
    tiered, _ram, disk = _tiered(tmp_path)
    tiered.set("k", 42, dict(NOTEBOOK))

    assert tiered.persist_from_memory("k", rebuild_seconds=0.05) is False
    assert disk.get("k") == (None, None)


def test_only_a_notebook_value_is_considered(tmp_path):
    tiered, _ram, disk = _tiered(tmp_path)
    tiered.set("k", 42, {"execution_time": 0.02})

    assert tiered.persist_from_memory("k", rebuild_seconds=2.0) is False
    assert disk.get("k") == (None, None)


def test_nothing_to_do_for_an_entry_ram_does_not_hold_or_disk_already_has(tmp_path):
    tiered, _ram, _disk = _tiered(tmp_path)
    assert tiered.persist_from_memory("missing", rebuild_seconds=2.0) is False
    tiered.set("slow", 42, {**NOTEBOOK, "execution_time": 3.0})
    assert tiered.persist_from_memory("slow", rebuild_seconds=5.0) is False
