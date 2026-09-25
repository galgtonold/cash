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
from cash._paths import _normalize_for
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


def test_a_path_that_exists_but_cannot_be_statted_is_never_fresh(tmp_path):
    """Snapshot time: an error other than "not there" (here a name too long
    for the file system) is recorded, not dropped."""
    too_long = str(tmp_path / ("x" * 300))
    snap = snapshot_dependencies({too_long})
    assert snap == {too_long: {"unresolved": True}}
    _, fresh, reason = dep_is_fresh(too_long, snap[too_long])
    assert (fresh, reason) == (False, "unresolved")


def test_a_file_the_call_deleted_is_still_dropped(tmp_path):
    """A temporary file the call wrote, read and removed is not a dependency
    that blocks every later hit."""
    gone = str(tmp_path / "scratch.tmp")
    assert snapshot_dependencies({gone}) == {}
