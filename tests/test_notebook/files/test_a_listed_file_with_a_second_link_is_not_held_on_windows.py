"""A file a listing found is not held when it has a second hard link.

On Windows the upstream check takes the stats of a large set of files from
folder listings (`file_dep_snapshot.stats_from_listings`), and a listing's
stat has no link count (``st_nlink`` 0). The folder watch kept such a stat
as that of a file with one link, so an edit made through the second link,
in a folder it does not watch (a dataset cache beside the project), went
unseen, and a cell below the read was served the old result. The watch now
asks one stat for the count, remembered until the folder reports a change.

Here a listing on Linux is made to look like a Windows one.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.upstream import statement_lineage
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

#: Past what is held at all (`statement_lineage._HOLD_MIN_FILES`).
FILES = 300


@pytest.fixture(autouse=True)
def _runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    monkeypatch.setattr(statement_lineage, "_HELD_FILE_KEYS", run_memo.HeldPaths(kept=8))
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_watch.reset()


@pytest.fixture(autouse=True)
def _listed_as_on_windows(monkeypatch):
    """Listings on every system, each stat with no link count."""

    def listed(paths):
        found = file_dep_snapshot.listing_stats(paths)
        return {p: _no_link_count(st) for p, st in found.items()}

    monkeypatch.setattr(run_memo, "stats_from_listings", listed)


def _no_link_count(st: os.stat_result) -> os.stat_result:
    extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
    return os.stat_result((*st[:3], 0, *st[4:]), extra)


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _photos(tmp_path, link: bool) -> list[str]:
    photos, cache = tmp_path / "photos", tmp_path / "cache"
    photos.mkdir()
    cache.mkdir()
    found = []
    for i in range(FILES):
        p = photos / f"p{i:03d}.pgm"
        if link:
            original = cache / p.name
            original.write_text(f"photo {i:03d}", encoding="utf-8")
            try:
                os.link(original, p)
            except (OSError, NotImplementedError):
                pytest.skip("no hard links here")
        else:
            p.write_text(f"photo {i:03d}", encoding="utf-8")
        found.append(str(p))
    return found


def _keys(paths: list[str]) -> dict:
    _present, keys = statement_lineage._present_file_keys(set(paths))
    _next_run()
    return dict(keys)


def test_an_edit_through_the_other_link_is_seen(tmp_path):
    assert run_memo.stats_from_listings is not file_dep_snapshot.stats_from_listings
    linked = _photos(tmp_path, link=True)
    if file_watch.watcher() is None:
        pytest.skip("no folder watch")
    for _ in range(3):
        _keys(linked)
    assert not statement_lineage._HELD_FILE_KEYS._entries, "a linked file was held"

    other = tmp_path / "cache" / "p007.pgm"
    st = os.stat(other)
    other.write_text("photo 007 EDITED", encoding="utf-8")
    os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    assert _keys(linked)[linked[7]] == stat_key(os.stat(linked[7]))


def test_files_with_one_link_are_still_held(tmp_path):
    single = _photos(tmp_path, link=False)
    if file_watch.watcher() is None:
        pytest.skip("no folder watch")
    for _ in range(3):
        assert _keys(single) == {p: stat_key(os.stat(p)) for p in single}
    assert statement_lineage._HELD_FILE_KEYS._entries, "files with one link were not held"
