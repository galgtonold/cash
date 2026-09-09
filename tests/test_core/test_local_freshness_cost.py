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


def test_the_second_check_of_an_unchanged_file_does_not_rehash(cash_instance, tmp_path):
    """The memo, asserted where it is decided rather than by a stopwatch."""
    path = _aged_file(tmp_path)
    runs: list[str] = []
    read = _reader(cash_instance, runs, path)

    read("a")                                   # cold: computes and snapshots
    file_dep_snapshot._HASH_MEMO.clear()
    read("a")                                   # warm: hashes once, memoizes
    memo_after_first = dict(file_dep_snapshot._HASH_MEMO)
    assert memo_after_first, "nothing was memoized"

    stamps = {k: v[0] for k, v in memo_after_first.items()}
    read("a")                                   # warm again: must reuse

    assert {k: v[0] for k, v in file_dep_snapshot._HASH_MEMO.items()} == stamps, (
        "the digest was recomputed for a file nothing had touched"
    )
    assert len(runs) == 1


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
