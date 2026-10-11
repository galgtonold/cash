"""``makedirs(d, exist_ok=True)`` on a folder that is already there makes nothing.

The ``os.mkdir`` audit event fires before the syscall, so the folder was taken
for one the call made, and every file read under it afterwards for the call's
own output: the reads were dropped, and an edit to the data kept serving the old
result. ``Path(out).parent.mkdir(exist_ok=True)`` for an output in the working
directory is ``mkdir(".")``, which dropped every input under the working
directory.
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = pytest.mark.core


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "x.txt"
    path.write_text("1", encoding="utf-8")
    return path


def test_makedirs_exist_ok_keeps_the_reads_under_it(disk_cash, data):
    @disk_cash.cache
    def load():
        os.makedirs(data.parent, exist_ok=True)  # @cash:assume-safe
        return data.read_text(encoding="utf-8")

    assert load() == "1"
    data.write_text("2", encoding="utf-8")
    assert load() == "2"


def test_path_mkdir_exist_ok_keeps_the_reads_under_it(disk_cash, data):
    @disk_cash.cache
    def load():
        pathlib.Path(data.parent).mkdir(parents=True, exist_ok=True)  # @cash:assume-safe
        return data.read_text(encoding="utf-8")

    assert load() == "1"
    data.write_text("2", encoding="utf-8")
    assert load() == "2"


def test_the_output_folder_being_the_working_directory_keeps_inputs(disk_cash, data, monkeypatch):
    monkeypatch.chdir(data.parent.parent)

    @disk_cash.cache
    def summarize(src, out):
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)  # @cash:assume-safe
        with open(src, encoding="utf-8") as fh:
            return fh.read()

    assert summarize("data/x.txt", "summary.txt") == "1"
    data.write_text("2", encoding="utf-8")
    assert summarize("data/x.txt", "summary.txt") == "2"


def test_a_new_file_in_a_listed_folder_that_existed_counts(disk_cash, data):
    @disk_cash.cache
    def names():
        os.makedirs(data.parent, exist_ok=True)  # @cash:assume-safe
        return sorted(os.listdir(data.parent))

    assert names() == ["x.txt"]
    (data.parent / "y.txt").write_text("3", encoding="utf-8")
    assert names() == ["x.txt", "y.txt"]


def test_a_folder_the_call_makes_still_holds_its_own_output(disk_cash, tmp_path):
    """Control: a folder that was not there is the call's: what it writes there
    and reads back is no input, so the call caches."""
    scratch = tmp_path / "scratch"

    @disk_cash.cache
    def round_trip(text):
        os.makedirs(scratch, exist_ok=True)  # @cash:assume-safe
        (scratch / "t.txt").write_text(text, encoding="utf-8")  # @cash:assume-safe
        out = (scratch / "t.txt").read_text(encoding="utf-8")
        (scratch / "t.txt").unlink()  # @cash:assume-safe
        scratch.rmdir()  # @cash:assume-safe
        return out

    assert round_trip("a") == "a"
    assert round_trip("a") == "a"
    assert round_trip.cache_info()["hits"] == 1
