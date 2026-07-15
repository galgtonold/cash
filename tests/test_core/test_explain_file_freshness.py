"""CAS-127: explain() must report file freshness the way a real lookup decides it.

CAS-98/CAS-10 (and CAS-119 for the decorator path) made file-dependency
freshness content-authoritative via the shared ``file_dep_is_fresh`` helper: a
touch (new mtime, identical bytes) is a HIT. ``_explain_call`` was never
migrated and still compared raw mtime/size, so after a touch it reported
``file_changed`` / ``'mtime changed'`` while the call itself hit the cache.

``explain()`` is what users reach for when they are already confused, so a
diagnostic that contradicts the behavior it describes is worse than none.
These tests pin explain() to the actual outcome in both directions.
"""
import os

import pytest

from cash import Cash


@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("hello world", encoding="utf-8")
    return p


@pytest.fixture
def reader(data_file):
    """A cached reader plus a call counter, so explain() can be checked
    against what the call ACTUALLY does rather than against itself."""
    c = Cash()
    calls = {"n": 0}

    @c.cache
    def read_it():
        calls["n"] += 1
        with open(data_file, encoding="utf-8") as fh:
            return fh.read()

    read_it()                       # prime: records the file dep
    assert calls["n"] == 1
    return read_it, calls


def _touch(path, delta=100):
    """Bump mtime without changing a single byte."""
    st = os.stat(path)
    os.utime(path, (st.st_atime + delta, st.st_mtime + delta))


class TestExplainFileFreshness:
    def test_touch_identical_content_explains_as_hit(self, reader, data_file):
        read_it, calls = reader
        _touch(data_file)

        e = read_it.explain()
        assert e.would_hit is True
        assert e.reason == "hit", (
            f"explain() reported {e.reason!r} ({e.details}) after a touch"
        )

        # ...and the call agrees: no recompute.
        read_it()
        assert calls["n"] == 1

    def test_same_size_content_edit_explains_as_file_changed(self, reader, data_file):
        read_it, calls = reader
        # Same byte length, different content -- the CAS-10 direction.
        data_file.write_text("HELLO WORLD", encoding="utf-8")
        assert os.path.getsize(data_file) == 11

        e = read_it.explain()
        assert e.would_hit is False
        assert e.reason == "file_changed"
        assert str(data_file) in {str(p) for p in e.details["changed_files"]} or any(
            p.endswith("data.txt") for p in e.details["changed_files"]
        )

        # ...and the call agrees: it recomputes.
        assert read_it() == "HELLO WORLD"
        assert calls["n"] == 2

    def test_size_change_explains_as_file_changed(self, reader, data_file):
        read_it, calls = reader
        data_file.write_text("hello world, and then some more", encoding="utf-8")

        e = read_it.explain()
        assert e.would_hit is False
        assert e.reason == "file_changed"
        assert any(
            v == "size changed" for v in e.details["changed_files"].values()
        ), e.details

        read_it()
        assert calls["n"] == 2

    def test_missing_file_explains_as_file_changed(self, reader, data_file):
        read_it, _ = reader
        os.remove(data_file)

        e = read_it.explain()
        assert e.would_hit is False
        assert e.reason == "file_changed"
        assert any(
            v == "file missing" for v in e.details["changed_files"].values()
        ), e.details

    def test_unchanged_file_explains_as_hit(self, reader):
        read_it, calls = reader
        e = read_it.explain()
        assert e.would_hit is True
        assert e.reason == "hit"
        read_it()
        assert calls["n"] == 1


class TestExplainAgreesWithCall:
    """The property the bug violated: explain() and the call never disagree."""

    @pytest.mark.parametrize(
        "mutate,expect_hit",
        [
            (lambda p: None, True),                                  # untouched
            (lambda p: _touch(p), True),                             # touch only
            (lambda p: p.write_text("HELLO WORLD", encoding="utf-8"), False),
            (lambda p: p.write_text("longer content here", encoding="utf-8"), False),
        ],
        ids=["untouched", "touched", "same_size_edit", "size_change"],
    )
    def test_explain_matches_actual_call(self, reader, data_file, mutate, expect_hit):
        read_it, calls = reader
        mutate(data_file)

        explained_hit = read_it.explain().would_hit
        before = calls["n"]
        read_it()
        actually_hit = calls["n"] == before

        assert explained_hit == actually_hit == expect_hit
