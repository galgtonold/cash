"""A file the call itself creates, reads back and removes is not an input.

Unzipping into a ``TemporaryDirectory``, writing a scratch file and reading it
back, or ``df.to_csv(tmp)`` then ``pd.read_csv(tmp)``: the read file was
recorded as an input, its removal before the call returned looked like "changed
while the call was running", and the function was never cached -- on every
call, in every process -- with a fix ("split the read and the write") that
cannot apply when a library does the write. The write into the scratch
directory was also reported as an effect a hit would skip.
"""

from __future__ import annotations

import os
import tempfile
import warnings
import zipfile

import pytest

from cash import Cash

pytestmark = pytest.mark.core


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _codes(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args)
    return [getattr(w.message, "code", None) for w in rec]


def _archive(tmp_path):
    archive = tmp_path / "in.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("x.txt", "hello")
    return str(archive)


def test_unzipping_into_a_temporary_directory_caches(c, tmp_path):
    archive = _archive(tmp_path)

    @c.cache
    def unzip_and_read(path):
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(path) as z:
                z.extractall(d)  # @cash:assume-safe
            with open(os.path.join(d, "x.txt"), encoding="utf-8") as fh:
                return fh.read()

    codes = _codes(unzip_and_read, archive)
    assert "STORE-INPUT-CHANGED" not in codes and "IMPURE-OBSERVED-EFFECTS" not in codes, codes
    assert unzip_and_read(archive) == "hello"
    assert unzip_and_read.cache_info()["hits"] == 1


def test_a_scratch_file_written_read_and_removed_caches(c, tmp_path):
    scratch = tmp_path / "scratch.txt"

    @c.cache
    def round_trip(text):
        with open(scratch, "w", encoding="utf-8") as fh:  # @cash:assume-safe
            fh.write(text.upper())
        with open(scratch, encoding="utf-8") as fh:
            out = fh.read()
        os.remove(scratch)  # @cash:assume-safe
        return out

    assert "STORE-INPUT-CHANGED" not in _codes(round_trip, "a")
    assert round_trip("a") == "A"
    assert round_trip.cache_info()["hits"] == 1


def test_a_mkstemp_file_caches(c):
    @c.cache
    def via_mkstemp(text):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(path, encoding="utf-8") as fh:
            out = fh.read()
        os.remove(path)  # @cash:assume-safe
        return out

    assert "STORE-INPUT-CHANGED" not in _codes(via_mkstemp, "a")
    assert via_mkstemp("a") == "a"
    assert via_mkstemp.cache_info()["hits"] == 1


def test_a_file_read_before_the_call_rewrites_it_is_still_an_input(c, tmp_path):
    """Control: read first, then rewritten -- which version was it computed from?"""
    data = tmp_path / "data.txt"
    data.write_text("1", encoding="utf-8")

    @c.cache
    def read_then_bump():
        value = data.read_text(encoding="utf-8")
        with open(data, "w", encoding="utf-8") as fh:  # @cash:assume-safe
            fh.write(value + "1")
        return value

    assert "STORE-INPUT-CHANGED" in _codes(read_then_bump)


def _observed(body):
    from cash.effect_observer import EffectObserver

    with EffectObserver() as observer:
        body()
    return [detail for kind, detail in observer.effects if kind == "file write"]


def test_a_write_into_a_scratch_directory_is_not_an_effect(tmp_path):
    def body():
        with (
            tempfile.TemporaryDirectory(dir=tmp_path) as d,
            open(os.path.join(d, "x.txt"), "w", encoding="utf-8") as fh,
        ):
            fh.write("x")

    assert _observed(body) == []


def test_a_write_into_a_directory_that_stays_is_still_an_effect(tmp_path):
    """Control: an output directory the call makes and leaves is a real effect."""
    out = tmp_path / "out"

    def body():
        os.makedirs(out)
        with open(out / "x.txt", "w", encoding="utf-8") as fh:
            fh.write("x")

    assert any("x.txt" in d for d in _observed(body))
