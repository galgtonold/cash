"""The RAM tier keeps big JSON-like values -- parsed records, an index of
lists -- in a form a hit reads back at C speed, and a hit is still a copy of
the value as it was stored.

A million nested records took 1.7 s to copy into the tier and 2.3 s to copy
out on every hit, a container at a time: more than building them. They are
now written once and read back in one pass. What a hit hands back must not
change with that: equal, independent of the stored entry and of the
original, every type as it was (a ``bool`` not an ``int``, ``-0.0``, a
``bytearray``, a date), the keys in their order, and a list the value holds
twice still one list. These pass before the change too: they guard it.
"""

from __future__ import annotations

import datetime
import math

import pytest

from cash.backends import FileBackend, InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

N = 5_000  # records: 15,000 containers, a big value for the tier


def _records(n: int = N) -> list:
    return [{"id": i, "tags": ["a", "b"], "meta": {"k": i, "even": i % 2 == 0}} for i in range(n)]


def _hit(value):
    backend = InMemoryBackend()
    backend.set("k", value)
    return backend, backend.get("k")[1]


def test_records_come_back_equal_and_independent():
    records = _records()
    backend, got = _hit(records)
    assert got == _records() and got is not records
    got[0]["tags"].append("caller's")
    got[1]["meta"]["k"] = "caller's"
    got.append("caller's")
    records[2]["tags"].append("the original's")
    assert backend.get("k")[1] == _records()


def test_every_hit_is_a_new_copy():
    backend, first = _hit(_records())
    second = backend.get("k")[1]
    assert first == second
    assert first is not second and first[0] is not second[0] and first[0]["tags"] is not second[0]["tags"]


def test_types_signs_and_key_order_are_kept():
    rows = [
        {
            "z": True,
            "a": 1,
            "neg": -0.0,
            "nan": math.nan,
            "c": 2j,
            "b": b"\x00",
            "big": 2**100,
            "s": "é\ud800",
            "n": None,
        }
        for _ in range(N)
    ]
    rows.append({"t": (1, [2], ()), "l": [], "d": {}})
    _backend, got = _hit(rows)
    first = got[0]
    assert list(first) == list(rows[0])
    assert type(first["z"]) is bool and type(first["a"]) is int
    assert math.copysign(1.0, first["neg"]) == -1.0 and math.isnan(first["nan"])
    assert first["c"] == 2j and first["b"] == b"\x00" and first["big"] == 2**100 and first["s"] == "é\ud800"
    assert got[-1] == {"t": (1, [2], ()), "l": [], "d": {}} and type(got[-1]["t"]) is tuple


def test_a_list_held_twice_inside_the_value_is_one_list_on_a_hit():
    common = ["x"]
    records = [{"id": i, "tags": common} for i in range(N)]
    backend, got = _hit(records)
    assert got[0]["tags"] is got[-1]["tags"] and got[0]["tags"] is not common
    got[0]["tags"].append("y")
    assert got[-1]["tags"] == ["x", "y"]
    assert backend.get("k")[1][-1]["tags"] == ["x"]


def test_a_bytearray_leaf_stays_a_bytearray_of_its_own():
    rows = [{"id": i, "raw": bytearray(b"ab")} for i in range(N)]
    backend, got = _hit(rows)
    assert type(got[0]["raw"]) is bytearray
    got[0]["raw"][0] = ord("z")
    assert backend.get("k")[1][0]["raw"] == bytearray(b"ab")


def test_dates_and_subclasses_keep_their_types():
    class Tag(str):
        pass

    day = datetime.date(2026, 10, 8)
    rows = [{"id": i, "day": day, "tags": [Tag("a")]} for i in range(N)]
    _backend, got = _hit(rows)
    assert type(got[0]["day"]) is datetime.date and got[0]["day"] == day
    assert type(got[0]["tags"][0]) is Tag


def test_an_entry_written_to_disk_after_the_store_is_the_value(tmp_path):
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "cache"))
    tiered = TieredBackend([ram, disk])
    notebook = {"execution_time": 0.02, "cost_model_family": "_GENERIC", "cost_model_type_name": "list"}
    tiered.set("k", {"variables": {"records": _records()}}, notebook)
    assert disk.get("k") == (None, None)

    assert tiered.persist_from_memory("k", rebuild_seconds=600.0) is True

    assert disk.get("k")[1] == {"variables": {"records": _records()}}
    meta, value = ram.peek_entry("k")
    assert value == {"variables": {"records": _records()}} and "DISK" in meta["storage"]
    assert ram.peek_entry("k", value=False) == (meta, None)


def test_records_beside_an_array_come_back_with_every_name_sharing_as_before():
    """A notebook entry with numpy's RNG state beside the variables is not
    JSON-like as a whole: the records in it are kept apart from the rest,
    and a tuple holding them still holds the very records the name does."""
    np = pytest.importorskip("numpy")
    records = _records()
    payload = {
        "variables": {"records": records, "same": records, "pair": (records, 1), "arr": np.arange(3)},
        "rng_state": {"numpy.random": np.random.get_state()},
    }
    backend, got = _hit(payload)
    variables = got["variables"]
    assert variables["records"] == _records() and variables["records"] is not records
    assert variables["same"] is variables["records"] and variables["pair"][0] is variables["records"]
    assert (variables["arr"] == np.arange(3)).all()
    variables["records"][0]["tags"].append("caller's")
    assert backend.get("k")[1]["variables"]["records"] == _records()


def test_a_list_inside_the_records_another_name_holds_stays_that_names_list():
    np = pytest.importorskip("numpy")
    records = _records()
    payload = {"variables": {"records": records, "tags": records[3]["tags"]}, "arr": np.arange(3)}
    _backend, got = _hit(payload)
    assert got["variables"]["tags"] is got["variables"]["records"][3]["tags"]
