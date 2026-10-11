"""A file the call creates and leaves behind is an input from then on.

A "download if missing, then read" loader creates its data file on the first
call and reads it on every later one. The read was dropped as the call's own
output, so the entry had no dependency on the file: replacing it kept serving
the first download's result, in later processes too, and deleting it did not
trigger a new download. A file the call removes before it returns (a scratch
file) is still no input (`test_a_file_the_call_creates_is_not_an_input`).
"""

from __future__ import annotations

import os
import shutil
import textwrap
import warnings

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core


def _codes(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args)
    return out, [getattr(w.message, "code", None) for w in rec]


@pytest.fixture
def remote(tmp_path):
    src = tmp_path / "remote.txt"
    src.write_text("1", encoding="utf-8")
    return src


def test_a_downloaded_file_replaced_later_is_read_again(disk_cash, tmp_path, remote):
    local = tmp_path / "data.txt"

    @disk_cash.cache
    def load(p):
        if not os.path.exists(p):
            shutil.copyfile(remote, p)  # @cash:assume-safe
        with open(p, encoding="utf-8") as fh:
            return fh.read()

    assert load(str(local)) == "1"
    assert load(str(local)) == "1"
    assert load.cache_info()["hits"] == 1, "an unchanged download must hit"
    local.write_text("22", encoding="utf-8")
    assert load(str(local)) == "22"


def test_a_downloaded_file_deleted_later_is_downloaded_again(disk_cash, tmp_path, remote):
    """Control: the probe that found it missing still answers for a deletion."""
    local = tmp_path / "data.txt"

    @disk_cash.cache
    def load(p):
        if not os.path.exists(p):
            shutil.copyfile(remote, p)  # @cash:assume-safe
        with open(p, encoding="utf-8") as fh:
            return fh.read()

    assert load(str(local)) == "1"
    local.unlink()
    remote.write_text("2", encoding="utf-8")
    assert load(str(local)) == "2"
    assert local.exists()


def test_a_file_written_read_back_and_appended_to_caches(disk_cash, tmp_path):
    """Control: what the call does to its own file while it runs is not
    "changed while the call was running"; the file as it is left is what the
    entry checks."""
    out = tmp_path / "log.txt"

    @disk_cash.cache
    def write_read_append(text):
        with open(out, "w", encoding="utf-8") as fh:  # @cash:assume-safe
            fh.write(text)
        with open(out, encoding="utf-8") as fh:
            got = fh.read()
        with open(out, "a", encoding="utf-8") as fh:  # @cash:assume-safe
            fh.write("!")
        return got

    value, codes = _codes(write_read_append, "a")
    assert value == "a" and "STORE-INPUT-CHANGED" not in codes, codes
    assert write_read_append("a") == "a"
    assert write_read_append.cache_info()["hits"] == 1


def test_a_listing_of_a_folder_the_call_made_and_kept_is_an_input(disk_cash, tmp_path):
    folder = tmp_path / "unpacked"

    @disk_cash.cache
    def names():
        if not folder.exists():
            folder.mkdir()  # @cash:assume-safe
            (folder / "a.txt").write_text("a", encoding="utf-8")  # @cash:assume-safe
        return sorted(os.listdir(folder))

    assert names() == ["a.txt"]
    assert names() == ["a.txt"]
    assert names.cache_info()["hits"] == 1
    (folder / "b.txt").write_text("b", encoding="utf-8")
    assert names() == ["a.txt", "b.txt"]


def test_a_replaced_download_is_seen_by_a_new_process(tmp_path, remote):
    (tmp_path / "main.py").write_text(
        textwrap.dedent(
            """
            import os, shutil, cash
            c = cash.Cash(cache_dir=".cash")

            @c.cache
            def load(p):
                if not os.path.exists(p):
                    shutil.copyfile("remote.txt", p)  # @cash:assume-safe
                with open(p) as fh:
                    return fh.read()

            print(load("data.txt"))
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "1"
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "1"
    (tmp_path / "data.txt").write_text("33", encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "33"
