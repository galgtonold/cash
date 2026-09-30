"""``os.stat``, ``os.path.getsize`` or ``os.listdir`` on a path that is not
there yet is an absent dependency, as ``open`` and ``os.path.exists`` are.

``try: os.stat('new.txt') except FileNotFoundError: return -1`` and the same
around ``os.listdir('newdir')`` kept serving the default after the file or
directory appeared: the failed call recorded nothing.
"""

from __future__ import annotations

import os

import pytest

from cash import Cash


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _via_stat(p):
    return os.stat(p).st_size


def _via_lstat(p):
    return os.lstat(p).st_size


def _via_getsize(p):
    return os.path.getsize(p)


@pytest.mark.parametrize("measure", [_via_stat, _via_lstat, _via_getsize])
def test_a_file_that_appears_after_a_failed_stat_recomputes(tmp_path, measure):
    c = _cash(tmp_path)
    target = tmp_path / "new.txt"

    @c.cache(assume_safe=True)
    def size_or_default(p):
        try:
            return measure(p)
        except FileNotFoundError:
            return -1

    assert size_or_default(str(target)) == -1
    assert size_or_default(str(target)) == -1
    target.write_bytes(b"abcd")
    assert size_or_default(str(target)) == 4


@pytest.mark.parametrize("lister", [os.listdir, lambda d: [e.name for e in os.scandir(d)]])
def test_a_directory_that_appears_after_a_failed_listing_recomputes(tmp_path, lister):
    c = _cash(tmp_path)
    target = tmp_path / "newdir"

    @c.cache(assume_safe=True)
    def names_or_default(d):
        try:
            return sorted(lister(d))
        except FileNotFoundError:
            return "missing"

    assert names_or_default(str(target)) == "missing"
    assert names_or_default(str(target)) == "missing"
    target.mkdir()
    assert names_or_default(str(target)) == []
