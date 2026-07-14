"""Content-hash fallback for file-dependency freshness (CAS-98, CAS-10).

The (mtime, size)-only freshness check fails two opposite ways:

* **CAS-98 (over-invalidation):** a touch-only change (identical content and
  size, only mtime bumped) makes the reader RECOMPUTE — it should stay a HIT.
* **CAS-10 (under-invalidation):** a real edit with the SAME size and a mtime
  the coarse check can't distinguish (sub-resolution / same-second edit) is
  MISSED — it should invalidate.

Recording a content hash at snapshot time and treating it as authoritative
when the size matches fixes both uniformly. These tests pin that behavior with
the real-kernel ``nb_runner`` oracle.

Ground truth: a ``read_csv`` statement's per-statement cache verdict is read
straight from cash's own debug trace (``%cash_debug on``) — ``Cache hit:
False`` for that statement means it recomputed, ``Cache hit: True`` means it
was restored. This is what a plain kernel vs a cached kernel differ on.
"""
from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(90)]

# content A and content B: DIFFERENT values, IDENTICAL byte length.
_CONTENT_A = b"val\n10\n20\n30\n"   # sum(val) == 60
_CONTENT_B = b"val\n40\n50\n60\n"   # sum(val) == 150
assert len(_CONTENT_A) == len(_CONTENT_B)


def _read_csv_recomputed(raw: str) -> bool:
    """Return True if the read_csv statement executed (cache miss), False if it
    was restored, by reading its verdict out of cash's debug trace."""
    marker = "Statement: df = pd.read_csv"
    idx = raw.find(marker)
    assert idx != -1, f"read_csv statement not found in debug trace: {raw[:400]!r}"
    tail = raw[idx:]
    hit_idx = tail.find("Cache hit:")
    assert hit_idx != -1, f"no cache verdict after read_csv statement: {tail[:400]!r}"
    return "Cache hit: False" in tail[hit_idx:hit_idx + 40]


def _reader_notebook(nb_runner, csv_str):
    nb_runner.create_notebook([
        "import pandas as pd\n"
        f"df = pd.read_csv('{csv_str}')\n"
        "print('sum =', int(df['val'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()


def test_cas98_touch_only_stays_cache_hit(nb_runner, tmp_path):
    """CAS-98: bumping only the mtime (identical content + size) must NOT
    invalidate — the reader restores from cache instead of recomputing."""
    csv = tmp_path / "data.csv"
    csv.write_bytes(_CONTENT_A)
    csv_str = str(csv).replace("\\", "/")

    _reader_notebook(nb_runner, csv_str)
    nb_runner.run_all()
    assert "sum = 60" in nb_runner.get_output(1)

    # Touch: bump mtime far into the future, content and size unchanged.
    m = os.stat(csv).st_mtime
    os.utime(csv, (m + 100, m + 100))
    assert csv.read_bytes() == _CONTENT_A          # content really is identical

    nb_runner.run_all()
    assert "sum = 60" in nb_runner.get_output(1)
    assert not _read_csv_recomputed(nb_runner.get_raw_output(1)), (
        "touch-only change must stay a cache hit, but read_csv recomputed"
    )


def test_cas10_same_size_quick_edit_invalidates(nb_runner, tmp_path):
    """CAS-10: a same-size edit under an mtime the coarse check can't tell
    apart must still invalidate — the reader re-executes with the new data."""
    csv = tmp_path / "data.csv"
    csv.write_bytes(_CONTENT_A)
    csv_str = str(csv).replace("\\", "/")

    _reader_notebook(nb_runner, csv_str)
    nb_runner.run_all()
    assert "sum = 60" in nb_runner.get_output(1)

    # Same-size overwrite, then force the mtime back to what the snapshot
    # recorded so (mtime, size) is provably ambiguous — content is the only
    # signal that distinguishes the two files. This deterministically stages
    # the sub-resolution / same-second edit CAS-10 describes.
    snapshot_mtime = os.stat(csv).st_mtime
    csv.write_bytes(_CONTENT_B)
    assert os.stat(csv).st_size == len(_CONTENT_A)   # same byte length
    os.utime(csv, (snapshot_mtime, snapshot_mtime))
    assert os.stat(csv).st_mtime == snapshot_mtime   # mtime is indistinguishable

    nb_runner.run_all()
    assert "sum = 150" in nb_runner.get_output(1), (
        "same-size edit with ambiguous mtime must re-execute the reader; "
        f"got: {nb_runner.get_output(1)!r}"
    )
    assert _read_csv_recomputed(nb_runner.get_raw_output(1)), (
        "content-different file must invalidate, but read_csv was restored"
    )
