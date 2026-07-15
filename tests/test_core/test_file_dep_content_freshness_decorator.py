"""Regression: ``@cash.cache`` file-dep freshness must be content-authoritative.

The notebook path got a content-hash freshness fallback in CAS-98/CAS-10; the
decorator path (``Cash._auto_file_deps_fresh``) still compared ``(mtime, size)``
only, so it failed in both opposite directions (CAS-119):

- **touch** (identical content + size, only the mtime bumped) spuriously
  invalidated and recomputed, and
- a **same-size edit** under an mtime the check can't tell apart (a back-to-back
  same-size overwrite yields an identical mtime on this filesystem) was missed
  and served stale data.

Both paths now route through the shared ``file_dep_is_fresh``: size first, then
CONTENT is authoritative whenever the size matches.
"""
from __future__ import annotations

import os

from cash import Cash, FileBackend


def _make_loader(cache_dir, calls):
    c = Cash(backend=FileBackend(cache_dir=str(cache_dir)))

    @c.cache
    def load(path):
        calls["n"] += 1
        with open(path) as f:
            return f.read()

    return load


def test_touch_only_change_still_hits(tmp_path):
    """CAS-98 direction: a touch must NOT invalidate.

    Identical bytes and size, only the mtime moved -> the entry is still fresh.
    The mtime-only baseline recomputed here (over-invalidation).
    """
    data = tmp_path / "data.txt"
    data.write_text("alpha-payload")
    calls = {"n": 0}
    load = _make_loader(tmp_path / "cache", calls)

    assert load(str(data)) == "alpha-payload"
    assert calls["n"] == 1

    # Touch: same bytes, same size, new mtime.
    st = os.stat(data)
    os.utime(data, (st.st_atime + 100, st.st_mtime + 100))
    assert os.stat(data).st_mtime != st.st_mtime, "test setup: mtime did not move"
    assert os.stat(data).st_size == st.st_size

    assert load(str(data)) == "alpha-payload"
    assert calls["n"] == 1, "touch-only change recomputed (over-invalidation)"


def test_same_size_edit_under_identical_mtime_invalidates(tmp_path):
    """CAS-10 direction: a same-size content change must invalidate.

    The mtime is forced back to the original so (mtime, size) is completely
    indistinguishable and only the content differs -- the mtime-only baseline
    served the stale cached string here.
    """
    data = tmp_path / "data.txt"
    data.write_text("alpha-payload")
    calls = {"n": 0}
    load = _make_loader(tmp_path / "cache", calls)

    assert load(str(data)) == "alpha-payload"
    assert calls["n"] == 1
    original = os.stat(data)

    # Same-size edit with the mtime pinned to its original value.
    data.write_text("bravo-payload")  # identical byte length
    os.utime(data, (original.st_atime, original.st_mtime))
    assert os.stat(data).st_size == original.st_size, "test setup: sizes differ"
    assert os.stat(data).st_mtime == original.st_mtime, "test setup: mtime moved"

    assert load(str(data)) == "bravo-payload", "served stale content after same-size edit"
    assert calls["n"] == 2, "same-size content change did not invalidate"


def test_unchanged_file_still_hits(tmp_path):
    """Control: an untouched file must stay a hit (no needless re-hash churn)."""
    data = tmp_path / "data.txt"
    data.write_text("alpha-payload")
    calls = {"n": 0}
    load = _make_loader(tmp_path / "cache", calls)

    assert load(str(data)) == "alpha-payload"
    assert load(str(data)) == "alpha-payload"
    assert calls["n"] == 1, "unchanged file recomputed"


def test_size_change_still_invalidates(tmp_path):
    """The cheap size-first check must keep catching plain content growth."""
    data = tmp_path / "data.txt"
    data.write_text("alpha")
    calls = {"n": 0}
    load = _make_loader(tmp_path / "cache", calls)

    assert load(str(data)) == "alpha"
    data.write_text("alpha-and-then-some-more")
    assert load(str(data)) == "alpha-and-then-some-more"
    assert calls["n"] == 2
