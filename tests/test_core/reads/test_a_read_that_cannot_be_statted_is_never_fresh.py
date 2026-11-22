"""A file a call read but cash cannot stat is a dependency that is never fresh.

On Windows a 295-character ``\\\\?\\C:\\...\\data.txt`` was recorded as
``//?/C:/...``: that spelling is not verbatim, the 260-character limit applied
again, ``os.stat`` failed, and the snapshot dropped the path. The entry was
stored with no dependency on the file, and every edit was served stale. The
spelling is fixed (verbatim prefixes keep their backslashes); and whatever
the spelling, a dependency that cannot be checked now fails closed.
"""

from __future__ import annotations

import os

import pytest

import cash.tracking.file_tracker as file_tracker
from cash import Cash
from cash._paths import _normalize_for, normalize_path
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import dep_is_fresh, snapshot_dependencies


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("\\\\?\\C:\\very\\long\\data.txt", "\\\\?\\C:\\very\\long\\data.txt"),
        ("//?/C:/very/long/data.txt", "\\\\?\\C:\\very\\long\\data.txt"),
        ("\\\\?\\UNC\\server\\share\\x.csv", "\\\\?\\UNC\\server\\share\\x.csv"),
        ("\\\\.\\C:\\x.csv", "\\\\.\\C:\\x.csv"),
        ("C:\\Users\\me\\x.csv", "C:/Users/me/x.csv"),
        ("\\\\server\\share\\x.csv", "//server/share/x.csv"),
    ],
)
def test_a_verbatim_path_keeps_its_backslashes_on_windows(path, expected):
    assert _normalize_for(path, "\\") == expected


def test_normalizing_is_a_no_op_on_posix():
    assert _normalize_for("/home/me/x.csv", "/") == "/home/me/x.csv"


def test_a_read_recorded_under_a_spelling_that_cannot_be_statted_recomputes(tmp_path, monkeypatch):
    """The Windows failure, reproduced on any platform: the recorded spelling
    of a read file is one ``os.stat`` rejects, as ``//?/C:/...`` was."""
    real = file_tracker.normalize_path
    monkeypatch.setattr(file_tracker, "normalize_path", lambda p: "//?/" + real(p) if os.path.isabs(p) else real(p))
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    data = tmp_path / "data.txt"
    data.write_text("v1", encoding="utf-8")
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def load(p):
        runs.append(1)
        with open(p, encoding="utf-8") as fh:
            return fh.read()

    assert load(str(data)) == "v1"
    data.write_text("v2", encoding="utf-8")
    assert load(str(data)) == "v2"
    assert len(runs) == 2


def test_a_call_that_read_a_file_it_cannot_stat_is_not_stored(tmp_path, monkeypatch):
    """An entry holding a dependency that is never fresh would be written on
    every call and never served: the call is not stored, and the next miss
    says so instead of "file changed". The file is one the snapshot cannot
    stat (a permission error), as it is stored.

    The read is recorded resolved and as ``normalize_path`` spells it, on
    Windows with forward slashes: the stand-in compares the file, not the
    spelling, or it never refuses there and the second call hits."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    data = tmp_path / "data.txt"
    data.write_text("v1", encoding="utf-8")
    real = file_dep_snapshot.own_stat

    def same_file(p) -> bool:
        return os.path.normcase(os.path.realpath(os.fspath(p))) == os.path.normcase(os.path.realpath(data))

    def refused(p, *a, **k):
        if same_file(p):
            raise PermissionError(13, "Permission denied", str(p))
        return real(p, *a, **k)

    monkeypatch.setattr(file_dep_snapshot, "own_stat", refused)

    @c.cache(assume_safe=True)
    def load(p):
        with open(p, encoding="utf-8") as fh:
            return fh.read()

    assert load(str(data)) == "v1"
    assert load(str(data)) == "v1"
    reasons = load.cache_info()["miss_reasons"]
    assert reasons.get("not stored last time") == 1, reasons
    why = str(load.explain(str(data)))
    assert f"not stored last time: it read {normalize_path(os.path.realpath(data))}, which cash cannot check" in why, (
        why
    )


def test_a_path_that_exists_but_cannot_be_statted_is_never_fresh(tmp_path):
    """Snapshot time: an error other than "not there" (here a name too long
    for the file system, or on Windows a name it refuses as malformed) is
    recorded, not dropped. Windows reports a too-long name as not there."""
    too_long = str(tmp_path / ("a<b.txt" if os.name == "nt" else "x" * 300))
    snap = snapshot_dependencies({too_long})
    assert snap == {too_long: {"unresolved": True}}
    _, fresh, reason = dep_is_fresh(too_long, snap[too_long])
    assert (fresh, reason) == (False, "unresolved")


def test_a_file_the_call_deleted_is_still_dropped(tmp_path):
    """A temporary file the call wrote, read and removed is not a dependency
    that blocks every later hit."""
    gone = str(tmp_path / "scratch.tmp")
    assert snapshot_dependencies({gone}) == {}
