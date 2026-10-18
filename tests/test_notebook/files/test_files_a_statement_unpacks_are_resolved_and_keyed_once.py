"""The files a statement unpacks and reads back cost one resolve and one stat each.

``fetch_lfw_people`` unpacks 13,000 images with ``tarfile`` into 3,400
folders and reads each one back with Pillow. Cash paid for each image:

* a full ``realpath`` at its open for writing, the file not being there yet
  (only a file that is there was resolved through its folder);
* its stat, resolve and version twice in the statement's lineage, once per
  spelling the read was recorded under (relative and absolute);
* an attempt to hash each folder, which ``io.FileIO`` refuses.

The lineage is the same bytes as before: each spelling still adds its part.
"""

from __future__ import annotations

import hashlib
import os
import sys

import pytest

from cash._paths import normalize_path
from cash.notebook.statement import file_deps
from cash.notebook.statement.file_deps import compute_file_hash_component
from cash.tracking import file_dep_snapshot, file_versions


@pytest.fixture
def in_a_run():
    file_dep_snapshot.begin_file_state_epoch()
    try:
        yield
    finally:
        file_dep_snapshot.end_file_state_epoch()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_file_not_there_yet_is_resolved_through_its_folder(tmp_path, monkeypatch, in_a_run):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "link")
    resolved = []
    real = os.path.realpath
    monkeypatch.setattr(
        file_dep_snapshot.os.path, "realpath", lambda p, *a, **k: resolved.append(p) or real(p, *a, **k)
    )
    first = file_dep_snapshot.realpath_of_read_this_run(str(tmp_path / "link" / "a.jpg"))
    second = file_dep_snapshot.realpath_of_read_this_run(str(tmp_path / "link" / "b.jpg"))
    assert first == (real(str(tmp_path / "link" / "a.jpg")), None)
    assert second == (real(str(tmp_path / "link" / "b.jpg")), None)
    assert resolved == [str(tmp_path / "link")], "the folder once, neither file in full"


def test_a_folder_is_not_opened_to_be_hashed(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(file_dep_snapshot, "_read_digest", lambda p, size: opened.append(p))
    assert file_dep_snapshot.file_content_hash(str(tmp_path)) is None
    assert not opened


def test_both_spellings_of_a_read_are_keyed_as_before_from_one_look(tmp_path, monkeypatch):
    monkeypatch.setattr(file_versions, "_STARTED_NS", 1 << 62)  # not written by this process
    monkeypatch.setattr(file_deps, "get_notebook_path", lambda: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "faces").mkdir()
    image = tmp_path / "faces" / "a.jpg"
    image.write_bytes(b"jpg")
    looked = []
    real = file_deps.realpath_of_read_this_run
    monkeypatch.setattr(file_deps, "realpath_of_read_this_run", lambda p: looked.append(p) or real(p))

    component = compute_file_hash_component({str(image), os.path.join("faces", "a.jpg")})

    st = os.stat(image)
    part = f"{normalize_path(os.path.realpath(image))}:{st.st_mtime_ns}:{st.st_size}:{st.st_ctime_ns}"
    assert component == ":" + hashlib.sha256(f"{part},{part}".encode()).hexdigest()
    assert looked == [str(image)]
