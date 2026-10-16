"""A file rewritten while this process runs is named by its content.

A statement that writes a file and then reads it (``extractall`` then
``read_csv``) is never stored, so no content check links the file's new stat
to the old one, and every reader below keyed apart after a restart. A file
written since the process started is hashed once per stat instead: the same
bytes keep the earliest version, new bytes are a version of their own.
"""

from __future__ import annotations

import os

from cash._paths import normalize_path
from cash.tracking import file_versions
from cash.tracking.file_dep_snapshot import file_content_hash


def _version(path):
    st = os.stat(path)
    canonical = normalize_path(os.path.realpath(path))
    return file_versions.content_version(canonical, st, lambda: file_content_hash(str(path), st.st_size, st))


def _rewrite(path, text, bump):
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump))


def test_the_same_bytes_keep_the_first_version(tmp_path):
    data = tmp_path / "rows.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    first = _version(data)
    _rewrite(data, "a,b\n1,2\n", 3_000_000_000)
    assert _version(data) == first


def test_new_bytes_are_a_new_version(tmp_path):
    data = tmp_path / "rows.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    first = _version(data)
    _rewrite(data, "a,b\n1,3\n", 3_000_000_000)
    assert _version(data) != first


def test_a_file_untouched_since_the_start_is_not_hashed(tmp_path, monkeypatch):
    data = tmp_path / "rows.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    old = file_versions._STARTED_NS - 10_000_000_000
    os.utime(data, ns=(old, old))
    if os.stat(data).st_ctime_ns >= file_versions._STARTED_NS:
        monkeypatch.setattr(file_versions, "_STARTED_NS", os.stat(data).st_ctime_ns + 1)
    asked = []
    st = os.stat(data)
    canonical = normalize_path(os.path.realpath(data))
    version = file_versions.content_version(canonical, st, lambda: asked.append(1) or "x")
    assert asked == []
    assert version == (st.st_mtime_ns, st.st_size, st.st_ctime_ns)
