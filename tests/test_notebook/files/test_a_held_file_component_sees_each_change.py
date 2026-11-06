"""A file component held under the folder watch still sees each change.

The file component of a lineage (`run_memo.file_component_this_run`) is
kept across cells while the folder watch vouches for its files, and a
folder that reports a change has only its own files looked at again: the
same stats give the same component. Each test changes a file of such a
folder in a way the component must follow, and checks it does. They pass
without the held component too: it was built again from every file.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.statement.file_deps import compute_file_hash_component
from cash.tracking import file_dep_snapshot, file_watch

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FOLDERS, FILES = 3, 10


@pytest.fixture(autouse=True)
def _runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._COVERED_COMPONENTS.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._COVERED_COMPONENTS.clear()
    file_watch.reset()


def _component(paths: set[str]) -> str:
    """One cell's component of *paths*, then the next cell run begins."""
    answer = run_memo.file_component_this_run(paths)
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()
    return answer


def _settled() -> None:
    """Take the watch's reports of what the test wrote (Windows may report
    a write a little later)."""
    file_watch.watcher().drain()


@pytest.fixture
def paths(tmp_path) -> set[str]:
    """Files in a few folders, half by a relative path, and one path not
    there; held by two components."""
    found = set()
    for d in range(FOLDERS):
        folder = tmp_path / f"set{d}"
        folder.mkdir()
        for i in range(FILES):
            p = folder / f"f{i:02d}.txt"
            p.write_text(f"file {d} {i:02d}", encoding="utf-8")
            found.add(f"set{d}/f{i:02d}.txt" if i % 2 else str(p))
    found.add(str(tmp_path / "set0" / "later.txt"))
    first = _component(found)
    assert _component(found) == first
    assert run_memo._COVERED_COMPONENTS, "no component was held: the tests below would prove nothing"
    return found


def _fresh(paths: set[str]) -> str:
    """The component built from every file, as without the held one."""
    return compute_file_hash_component(paths)


def test_a_save_beside_the_files_keeps_the_component(paths, tmp_path):
    before = _component(paths)
    (tmp_path / "set1" / "notes.ipynb").write_text("{}", encoding="utf-8")
    _settled()
    assert _component(paths) == before == _fresh(paths)


def test_an_edit_in_a_folder_that_reported_a_change_before_is_seen(paths, tmp_path):
    before = _component(paths)
    (tmp_path / "set1" / "notes.ipynb").write_text("{}", encoding="utf-8")
    _settled()
    assert _component(paths) == before
    p = tmp_path / "set1" / "f03.txt"
    st = os.stat(p)
    p.write_text("FILE 1 03", encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _settled()
    after = _component(paths)
    assert after != before
    assert after == _fresh(paths)


def test_a_deleted_file_is_seen(paths, tmp_path):
    before = _component(paths)
    (tmp_path / "set2" / "f04.txt").unlink()
    _settled()
    after = _component(paths)
    assert after != before
    assert after == _fresh(paths)


def test_a_file_that_appears_is_seen(paths, tmp_path):
    before = _component(paths)
    (tmp_path / "set0" / "later.txt").write_text("here now", encoding="utf-8")
    _settled()
    after = _component(paths)
    assert after != before
    assert after == _fresh(paths)


def test_a_file_replaced_by_a_link_is_seen(paths, tmp_path):
    if sys.platform == "win32":
        pytest.skip("making a link needs a privilege on Windows")
    before = _component(paths)
    target = tmp_path / "elsewhere.txt"
    target.write_text("another file", encoding="utf-8")
    p = tmp_path / "set2" / "f06.txt"
    p.unlink()
    p.symlink_to(target)
    _settled()
    after = _component(paths)
    assert after != before
    assert after == _fresh(paths)
