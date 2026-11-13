"""A stat the folder watch keeps is found whichever separator asks for it.

On Windows the run records a path by its portable spelling
(``C:/Users/me/people/p00``) and the system hands back the native one
(``C:\\Users\\me\\people\\p00``). The watch kept a folder's stat under the
spelling it was taken by, so asked by the other it held none: each of 40
folders was looked at again (two ``lstat`` calls) before the first cell
after it was listed. Simulated here on any system by making ``\\`` the
watch's second separator, as ``/`` is on Windows.
"""

from __future__ import annotations

import os

import pytest

from cash.tracking import file_watch


class _Backend:
    """Watches every folder and reports no change."""

    def add(self, path):
        return path

    def remove(self, key):
        pass

    def poll(self, sync=False):
        return []


def _other(path: str) -> str:
    return path.replace("/", "\\")


@pytest.fixture
def watch(monkeypatch):
    monkeypatch.setattr(file_watch, "_ALTSEP", "\\")
    monkeypatch.setattr(file_watch, "_SEP", "/")
    return file_watch.FileWatch(_Backend())


@pytest.fixture
def lstats(monkeypatch) -> list[str]:
    taken: list[str] = []
    real = file_watch.own_lstat

    def own_lstat(path):
        taken.append(path)
        return real(path)

    monkeypatch.setattr(file_watch, "own_lstat", own_lstat)
    return taken


@pytest.fixture
def folder(tmp_path) -> str:
    if "\\" in str(tmp_path):
        pytest.skip("the temporary folder's path holds a backslash")
    (tmp_path / "people" / "p00").mkdir(parents=True)
    (tmp_path / "people" / "p00" / "a.jpg").write_bytes(b"x")
    return str(tmp_path / "people" / "p00")


def test_a_folder_taken_from_a_listing_is_found_by_the_other_spelling(watch, lstats, folder):
    parent = os.path.dirname(folder)
    token = watch.before_listing(_other(parent))
    assert token is not None, "the folder's other spelling was not watched"
    watch.keep_listing(token, {os.path.normcase("p00"): os.lstat(folder)})
    st = watch.take_listed(folder)
    assert st is not None

    lstats.clear()
    assert watch.take_listed(_other(folder)) is st
    assert watch.current(_other(folder)) is st
    assert watch.current(folder) is st
    assert watch.cover([folder, _other(folder)]) is not None
    assert lstats == [], "the folder was looked at again under its other spelling"


def test_a_file_kept_by_one_spelling_is_current_by_the_other(watch, folder):
    path = os.path.join(folder, "a.jpg")
    token = watch.before_stat(_other(path))
    assert token is not None
    st = os.lstat(path)
    watch.record(token, _other(path), st)
    assert watch.current(path) is st
    assert watch.current(_other(path)) is st
    parts, loose = watch.cover_by_folder({"a": path, "b": _other(path)})
    assert loose == [] and sorted(parts[0][2]) == ["a", "b"]


def test_a_verbatim_path_keeps_its_separators(monkeypatch):
    monkeypatch.setattr(file_watch, "_ALTSEP", "/")
    monkeypatch.setattr(file_watch, "_SEP", "\\")
    assert file_watch.watch_spelling("C:/data/p00") == "C:\\data\\p00"
    assert file_watch.watch_spelling("\\\\?\\C:\\data/p00") == "\\\\?\\C:\\data/p00"


def test_on_posix_a_path_is_kept_as_given():
    if os.altsep is not None:
        pytest.skip("a system with a second separator")
    assert file_watch.watch_spelling("/data\\p00") == "/data\\p00"
