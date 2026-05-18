from pathlib import Path

import pytest

from benchmarks._ser_deser_measure import MeasureResult, measure_one


def test_measure_one_returns_valid_result_for_inmemory(tmp_path):
    r = measure_one(
        family="dict_shallow",
        target_bytes=10_000,
        backend_kind="ram",
        repeats=2,
        cache_root=tmp_path,
    )
    assert isinstance(r, MeasureResult)
    assert r.family == "dict_shallow"
    assert r.backend_kind == "ram"
    assert r.target_bytes == 10_000
    assert r.actual_size_bytes > 0
    assert r.serialize_seconds >= 0
    assert r.deserialize_seconds >= 0
    assert r.repeats == 2
    assert r.error is None


def test_measure_one_returns_valid_result_for_disk(tmp_path):
    r = measure_one(
        family="bytes",
        target_bytes=1_000,
        backend_kind="disk",
        repeats=2,
        cache_root=tmp_path,
    )
    assert r.backend_kind == "disk"
    assert r.error is None
    # Disk roundtrip writes a real file.
    assert any(tmp_path.rglob("*"))


def test_measure_one_handles_missing_optional_dep_gracefully(tmp_path, monkeypatch):
    """If sparse generator raises ImportError, measure_one returns a result
    with error set, not an exception."""
    from benchmarks import _object_generators

    def boom(target_bytes):
        raise ImportError("scipy unavailable")

    monkeypatch.setitem(_object_generators._GENERATORS, "sparse", boom)
    r = measure_one(
        family="sparse",
        target_bytes=10_000,
        backend_kind="ram",
        repeats=2,
        cache_root=tmp_path,
    )
    assert r.error is not None and "scipy unavailable" in r.error
    assert r.serialize_seconds == 0.0
    assert r.deserialize_seconds == 0.0
