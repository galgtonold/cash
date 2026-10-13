"""The bytes ceiling weighs a frame by what it writes to disk, not by its RAM.

The ceiling asks whether the disk an entry takes is worth the compute it
saves, and it read the RAM tier's size for the disk's. A column of Python
strings is about 50 bytes of object header per row in RAM and a few bytes
in a pickle: a retail notebook's feature frame was 445 MiB in RAM and 233 MiB
written. Judged by the first, its 3.4 s rebuild came out at 131 MiB/s
against the 128 MiB/s ceiling, so it was kept on one run and refused on the
next as the seconds wavered, and after a restart it was rebuilt, with the
whole chain before it. The warning even called the RAM size "serialized".
"""

from __future__ import annotations

import pickle
import warnings

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

MIB = 1024**2
ROWS = 1_200_000


def _strings_frame():
    """~70 MiB in RAM, ~11 MiB pickled: four region names, each held by a
    quarter of the rows, written once each and then as memo references."""
    names = np.array(["north", "south", "east", "west"], dtype=object)
    return pd.DataFrame({"region": pd.Series(names[np.arange(ROWS) % 4], dtype=object), "units": np.arange(ROWS)})


def _numbers_frame():
    """~73 MiB in RAM and on disk alike."""
    return pd.DataFrame({"a": np.arange(ROWS * 4, dtype="int64"), "b": np.zeros(ROWS * 4)})


def _store(tmp_path, frame, seconds):
    """The caller's metadata, which the tiers write back to, beside the RAM
    tier's own, and the refusals said."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    meta = {"execution_time": seconds, "code": "feat = f(df)"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.set("stmt:feat", {"variables": {"feat": frame}}, meta)
        backend.shutdown()
    in_ram = backend.backends[0].peek_entry("stmt:feat", value=False)[0]
    said = [str(w.message) for w in caught if "CACHE-NOT-WORTH-BYTES" in str(w.message)]
    return meta, in_ram, said


def test_a_frame_of_strings_is_weighed_by_its_pickle(tmp_path, monkeypatch):
    # Only the bytes ceiling is under test: on a busy machine the timed
    # restore check refused the 0.5 s entry before the ceiling weighed it.
    from cash.backends import persistence_policy

    monkeypatch.setattr(persistence_policy.PersistencePolicy, "pays_to_restore", lambda *a, **k: True)
    monkeypatch.setattr(persistence_policy, "_write_outweighs_restore", lambda *a, **k: False)
    frame = _strings_frame()
    written = len(pickle.dumps(frame, protocol=5))

    meta, in_ram, said = _store(tmp_path, frame, 0.5)

    assert in_ram["size"] > 128 * MIB * 0.5 > written  # refused by its RAM, kept by its pickle
    assert in_ram.get("serialized_bytes_estimate") == pytest.approx(written, rel=0.2)
    assert "DISK" in (meta.get("storage") or []), (meta, said)
    assert list(tmp_path.glob("*.entry"))
    assert not said


def test_a_frame_too_big_on_disk_too_is_still_refused(tmp_path):
    """Negative control: numbers take on disk what they take in RAM."""
    meta, _in_ram, said = _store(tmp_path, _numbers_frame(), 0.5)

    assert "DISK" not in (meta.get("storage") or []), meta
    assert meta.get("persist_skipped") == "bytes", meta
    assert not list(tmp_path.glob("*.entry"))
    assert len(said) == 1


def test_the_estimate_only_ever_lowers_the_weight():
    from cash.backends.persistence_policy import SERIALIZED_ESTIMATE, disk_bytes

    assert disk_bytes({}, 100) == 100
    assert disk_bytes({SERIALIZED_ESTIMATE: 40}, 100) == 40
    assert disk_bytes({SERIALIZED_ESTIMATE: 400}, 100) == 100
    assert disk_bytes({SERIALIZED_ESTIMATE: 0}, 100) == 100
