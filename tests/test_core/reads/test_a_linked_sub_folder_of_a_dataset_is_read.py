"""A dataset folder with a linked sub-folder depends on the files under the link.

pandas, pyarrow and polars read ``dd/sub -> ../other/sub`` as part of ``dd``;
cash expanded the folder with ``os.walk``, which does not follow links, so an
edit or a new file under the link kept serving the old total -- for an
auto-tracked ``pd.read_parquet("dd")``, ``file_depends_on="dd"`` and
``FileDataSource("dd")`` alike. A link back up the tree is walked once.
"""

from __future__ import annotations

import os

import pytest

from cash.file_source import FileDataSource
from cash.tracking.tracker_context import untracked

pytestmark = pytest.mark.core


def _link(target, link):
    """A directory symlink, or skip where the OS needs a privilege for one
    (Windows without developer mode)."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot make a directory symlink here: {exc}")


@pytest.fixture
def dataset(tmp_path):
    dd = tmp_path / "dd"
    dd.mkdir()
    (dd / "a.txt").write_text("1", encoding="utf-8")
    other = tmp_path / "other" / "sub"
    other.mkdir(parents=True)
    (other / "b.txt").write_text("10", encoding="utf-8")
    _link(other, dd / "sub")
    _settle(tmp_path)
    return dd, other


def _settle(top):
    """Put every timestamp an hour back: a folder changed a moment ago is
    re-checked on every lookup whatever it holds, which would hide the bug."""
    for root, dirs, files in os.walk(top):
        for name in dirs + files:
            path = os.path.join(root, name)
            if not os.path.islink(path):
                st = os.stat(path)
                os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))
    st = os.stat(top)
    os.utime(top, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _read_unseen(path) -> int:
    fd = os.open(path, os.O_RDONLY)  # raises no read event
    try:
        return int(os.read(fd, 64))
    finally:
        os.close(fd)


def _total(folder) -> int:
    """Read the folder as a tool in another process would: unseen, so the
    files reach the entry only through the declared folder."""
    with untracked():
        return sum(
            _read_unseen(os.path.join(root, name))
            for root, _dirs, files in os.walk(folder, followlinks=True)
            for name in files
        )


def test_file_depends_on_a_folder_sees_an_edit_under_a_link(disk_cash, dataset):
    dd, other = dataset
    runs = []

    @disk_cash.cache(file_depends_on=str(dd))
    def total():
        runs.append(1)
        return _total(dd)

    assert total() == 11
    assert total() == 11 and len(runs) == 1
    (other / "b.txt").write_text("20", encoding="utf-8")
    assert total() == 21


def test_file_depends_on_a_folder_sees_a_new_file_under_a_link(disk_cash, dataset):
    dd, other = dataset

    @disk_cash.cache(file_depends_on=str(dd))
    def total():
        return _total(dd)

    assert total() == 11
    (other / "c.txt").write_text("5", encoding="utf-8")
    assert total() == 16


def test_a_file_source_on_a_folder_follows_a_link(dataset):
    dd, other = dataset
    source = FileDataSource(str(dd))
    before = source.state_token()
    (other / "b.txt").write_text("20", encoding="utf-8")
    assert source.state_token() != before


def test_a_dataset_reader_on_a_folder_follows_a_link(disk_cash, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")
    dd = tmp_path / "dd"
    dd.mkdir()
    pq.write_table(pa.table({"x": [1]}), dd / "p1.parquet")
    other = tmp_path / "other" / "sub"
    other.mkdir(parents=True)
    pq.write_table(pa.table({"x": [10]}), other / "p2.parquet")
    _link(other, dd / "sub")
    _settle(tmp_path)

    @disk_cash.cache
    def total():
        return int(pd.read_parquet(dd).x.sum())

    assert total() == 11
    pq.write_table(pa.table({"x": [20]}), other / "p2.parquet")
    assert total() == 21


def test_a_link_back_up_the_tree_is_walked_once(tmp_path):
    from cash.tracking.file_dep_snapshot import walk_dataset_dir

    top = tmp_path / "dd"
    (top / "sub").mkdir(parents=True)
    (top / "sub" / "x.txt").write_text("1", encoding="utf-8")
    _link(top, top / "sub" / "loop")
    roots = [root for root, _dirs, _files in walk_dataset_dir(str(top))]
    assert len(roots) == 2, roots
    names = [name for _root, _dirs, files in walk_dataset_dir(str(top)) for name in files]
    assert names == ["x.txt"]
