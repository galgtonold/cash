"""explain() and the per-call log call a file "sampled" only when it is.

Every file snapshot records a ctime, so a label keyed on the ctime called
every small file "sampled ... trusted to its timestamps", and the HIT line
named a file opened by a relative path twice (relative and absolute).
"""

from __future__ import annotations

import logging
import os

import pytest

from cash import Cash, FileBackend

pytestmark = pytest.mark.core


def _reader(tmp_path, cap):
    c = Cash(
        backend=FileBackend(cache_dir=str(tmp_path / ".cash")),
        register_magic=False,
        verbose=True,
        file_hash_full_max_bytes=cap,
    )

    @c.cache(assume_safe=True)
    def load(path):
        with open(path, "rb") as fh:
            return len(fh.read())

    return load


def _hit_line(load, path, caplog):
    with caplog.at_level(logging.INFO, logger="cash.calls"):
        load(path)
        load(path)
    lines = [r.getMessage() for r in caplog.records if r.name == "cash.calls"]
    hits = [line for line in lines if line.startswith("HIT")]
    assert hits, lines
    return hits[-1]


def test_a_small_file_is_not_called_sampled(tmp_path, caplog):
    data = tmp_path / "data.csv"
    data.write_bytes(b"a,b\n1,2\n3,4\n")
    load = _reader(tmp_path, cap=1 << 20)

    line = _hit_line(load, str(data), caplog)
    assert "trusts the timestamps" not in line, line
    shown = load.explain(str(data)).details["file_deps"]
    assert shown, "the file was not recorded"
    assert all("sampled" not in desc for desc in shown.values()), shown
    assert any("hash" in desc for desc in shown.values()), shown


def test_a_large_file_is_called_sampled_once(tmp_path, caplog, monkeypatch):
    """Positive control: above the cap the label is right, and a file opened
    by a relative path is named once."""
    (tmp_path / "big.bin").write_bytes(os.urandom(5000))
    monkeypatch.chdir(tmp_path)
    load = _reader(tmp_path, cap=1000)

    line = _hit_line(load, "big.bin", caplog)
    assert "trusts the timestamps of big.bin (sampled" in line, line
    assert line.count("big.bin") == 1, line
    shown = load.explain("big.bin").details["file_deps"]
    assert len(shown) == 1 and "sampled hash" in next(iter(shown.values())), shown
