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


import csv
import subprocess
import sys


def test_matrix_cli_writes_csv_with_expected_columns(tmp_path):
    """Smoke test: run the matrix CLI on a tiny subset and check the CSV
    schema."""
    out_csv = tmp_path / "matrix.csv"
    cache_dir = tmp_path / "cache"
    cmd = [
        sys.executable,
        "benchmarks/measure_ser_deser.py",
        "--out", str(out_csv),
        "--cache-root", str(cache_dir),
        "--families", "dict_shallow,bytes",
        "--sizes", "1000,10000",
        "--backends", "ram",
        "--repeats", "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    assert out_csv.exists()
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 2 families × 2 sizes × 1 backend = 4 rows
    assert len(rows) == 4
    expected_cols = {
        "family", "target_bytes", "backend_kind", "repeats",
        "actual_size_bytes", "serialize_seconds", "deserialize_seconds", "error",
    }
    assert expected_cols.issubset(rows[0].keys())
    # All cells should have succeeded (no errors).
    assert all(r["error"] == "" for r in rows)
