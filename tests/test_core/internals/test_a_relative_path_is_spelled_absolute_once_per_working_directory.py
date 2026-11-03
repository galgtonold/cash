"""A relative path is spelled absolute once per working directory.

Each check of each snapshot spelled every relative dependency absolute again
(`spelled_absolute`): 361,000 times, and 517,000 ``abspath`` calls, before
one cell of a notebook over installed packages. The spelling is kept while
the working directory is the same.
"""

from __future__ import annotations

import os

from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import spelled_absolute


def test_a_spelling_is_made_once_and_again_in_another_directory(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    made = []
    real = os.path.abspath

    def counted(path):
        made.append(path)
        return real(path)

    monkeypatch.setattr(file_dep_snapshot.os.path, "abspath", counted)
    first = spelled_absolute("data/x.csv")
    for _ in range(100):
        assert spelled_absolute("data/x.csv") == first
    assert made == ["data/x.csv"]
    assert first.endswith("/a/data/x.csv")

    monkeypatch.chdir(tmp_path / "b")
    assert spelled_absolute("data/x.csv").endswith("/b/data/x.csv")
    assert spelled_absolute("../x.csv") is None
