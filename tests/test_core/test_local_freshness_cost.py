"""Checking your own files for freshness is work, and it is now measured.

Two things ship here, both from one question a round-16 session raised: a
pipeline over fifty inputs pays for fifty freshness checks on **every** cache
hit, and file dependencies propagate -- an aggregate that calls ten cached
functions inherits their inputs -- so ten hits pay it ten times.

* the digest is memoized per process, so the second and later checks of an
  unchanged file are a ``stat``. Measured on this box, warm page cache, median
  of five warm hits, against the round-16 wheel:

      50 files x  2 MiB    79.6 ms  ->  2.7 ms
      50 files x  8 MiB   313.3 ms  ->  2.8 ms
      50 files x 32 MiB    20.1 ms  ->  2.7 ms   (sampled either way)

* what remains is measured and reported. Cash already warned when checking
  REMOTE sources cost more than it saved; local hashing had no such guard, so a
  pipeline paying for it had nothing to look at. Same rule, same judgement, new
  code (``CACHE-FRESHNESS-COST``).

The memo's own guard rails are pinned in
``test_file_dep_content_freshness_decorator.py`` and
``test_sampled_file_freshness_backstops.py``: a file is memoized only once it
has been untouched for a while, and a digest is reused for a few seconds only,
so "write it then read it twice" and "raise the threshold to be certain" both
still behave.
"""
from __future__ import annotations

import os
import time
import warnings

import pytest

from cash import Cash
from cash.notebook import file_dep_snapshot


@pytest.fixture
def cash_instance(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _aged_file(tmp_path, name="input.csv", mib=1):
    """A tracked input old enough for the memo to consider it settled."""
    path = tmp_path / name
    with open(path, "wb") as fh:
        for _ in range(mib):
            fh.write(b"x" * (1024 * 1024))
    st = os.stat(path)
    os.utime(path, (st.st_atime - 3600, st.st_mtime - 3600))
    return str(path)


def _reader(c, runs, path):
    @c.cache(assume_safe=True)
    def read(tag):
        runs.append(tag)
        with open(path, "rb") as fh:
            return len(fh.read())

    return read


def test_checks_inside_one_call_share_a_digest(cash_instance, tmp_path, monkeypatch):
    """The memo, asserted where it is decided rather than by a stopwatch: an
    aggregate whose cached helpers all depend on one input hashes it once per
    call, not once per helper (the round-16 pipeline: fifty inputs, ten
    helpers)."""
    import hashlib
    import types

    path = _aged_file(tmp_path)
    runs: list[str] = []
    read = _reader(cash_instance, runs, path)

    @cash_instance.cache(assume_safe=True)
    def aggregate(n):
        return sum(read(str(i)) for i in range(n))

    aggregate(3)                                # cold: computes and snapshots
    aggregate.cache_clear()                     # the aggregate misses, its helpers hit
    hashed: list[int] = []
    monkeypatch.setattr(file_dep_snapshot, "hashlib", types.SimpleNamespace(
        sha256=lambda *a: hashed.append(1) or hashlib.sha256(*a)))
    aggregate(3)
    assert len(runs) == 3, "the helpers did not hit"
    assert len(hashed) == 1, f"one call hashed its one input {len(hashed)} times"


def test_each_call_rehashes_a_file_under_the_cap(cash_instance, tmp_path):
    """Round 20 (r20s5): a long-lived process served a stale result for a
    <=256 MiB file edited in place with its size and mtime unchanged (an
    np.memmap write; a write + os.utime back) -- the memo trusted the stat for
    five seconds. Under the cap, the content decides on every call."""
    path = _aged_file(tmp_path)
    runs: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def first_byte(tag):
        runs.append(tag)
        with open(path, "rb") as fh:
            return fh.read(1)

    assert first_byte("a") == b"x"
    assert first_byte("a") == b"x"             # warm: the digest is memoized now
    before = os.stat(path)
    with open(path, "r+b") as fh:              # same size, then the mtime put back
        fh.write(b"Z")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert os.stat(path).st_size == before.st_size

    assert first_byte("a") == b"Z", "served the old content for an edited file"
    assert len(runs) == 2


def test_an_edit_still_invalidates_with_the_memo_warm(cash_instance, tmp_path):
    """The control that matters: speed must not cost correctness."""
    path = _aged_file(tmp_path)
    runs: list[str] = []
    read = _reader(cash_instance, runs, path)

    assert read("a") == 1024 * 1024
    with open(path, "ab") as fh:                # ordinary edit: size and mtime move
        fh.write(b"more")

    assert read("a") == 1024 * 1024 + 4
    assert len(runs) == 2


def test_a_freshly_written_file_is_not_memoized(cash_instance, tmp_path):
    """A file written moments ago is the one that may still be changing."""
    path = str(tmp_path / "fresh.csv")
    with open(path, "wb") as fh:
        fh.write(b"y" * (1024 * 1024))

    file_dep_snapshot._HASH_MEMO.clear()
    file_dep_snapshot.file_content_hash(path)

    assert not file_dep_snapshot._HASH_MEMO, (
        "a file written a moment ago was memoized"
    )


# --------------------------------------------------------------------------- #
# The cost report                                                             #
# --------------------------------------------------------------------------- #

def test_expensive_local_validation_is_reported(cash_instance, tmp_path, monkeypatch):
    """Slow checking is now visible, with the same rule the remote path uses.

    The delay is injected rather than provoked: reaching the threshold for real
    needs hundreds of files or a slow filesystem, and neither belongs in a unit
    test. What is under test is that the cost is measured and judged at all --
    before this, only the remote channel was.
    """
    path = _aged_file(tmp_path)
    runs: list[str] = []
    read = _reader(cash_instance, runs, path)
    read("a")

    real = file_dep_snapshot.file_dep_is_fresh

    def slow(*args, **kwargs):
        time.sleep(0.4)
        return real(*args, **kwargs)

    monkeypatch.setattr(file_dep_snapshot, "file_dep_is_fresh", slow)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        read("a")

    text = "\n".join(str(w.message) for w in rec)
    assert "CACHE-FRESHNESS-COST" in text, f"a slow check said nothing:\n{text}"
    assert "1 tracked file" in text, text


def test_ordinary_validation_says_nothing(cash_instance, tmp_path):
    """The control: the warning must be rare enough to mean something."""
    path = _aged_file(tmp_path)
    runs: list[str] = []
    read = _reader(cash_instance, runs, path)
    read("a")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        read("a")
        read("a")

    assert "CACHE-FRESHNESS-COST" not in "\n".join(str(w.message) for w in rec)
