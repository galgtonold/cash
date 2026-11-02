"""Whether a writer rewrites a file its input was read from, without resolving every read.

A frame built by a loop over 24,000 documents carries all of them as the
files it was read from. Before each cell reading the frame, the check of a
writer below it (``df.to_csv(...)``) resolved each of those paths: 276,000
``lstat`` calls a cell, and on Windows, where resolving opens the file,
13 s. A crowded folder is now listed once, and only a path that could
resolve to the written file is resolved: one with its name, a link, or one
the listing does not hold. The answer is the one resolving them all gives.
"""

from __future__ import annotations

import os
import random

import pytest

from cash.notebook.upstream.file_writers import any_resolves_to

N = 40


def _resolving_each(paths, targets) -> bool:
    return any(os.path.realpath(p) in targets for p in paths)


def _crowd(folder, n=N) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"{i}_7.txt"
        p.write_text(f"review {i}\n", encoding="utf-8")
        paths.append(str(p))
    return paths


def _can_link(tmp_path) -> bool:
    try:
        os.symlink(tmp_path / "nowhere", tmp_path / "probe-link")
    except (OSError, NotImplementedError):
        return False
    os.unlink(tmp_path / "probe-link")
    return True


@pytest.fixture
def links(tmp_path):
    if not _can_link(tmp_path):
        pytest.skip("this system makes no symbolic links")


def test_documents_other_than_the_written_file_are_not_it(tmp_path):
    reads = _crowd(tmp_path / "pos")
    written = tmp_path / "movie_data.csv"
    written.write_text("review,sentiment\n", encoding="utf-8")

    assert any_resolves_to(reads, {os.path.realpath(written)}) is False


def test_the_written_file_among_the_documents_is_found(tmp_path):
    reads = _crowd(tmp_path / "pos")
    written = reads[17]

    assert any_resolves_to(reads, {os.path.realpath(written)}) is True


def test_a_file_of_the_same_name_elsewhere_is_not_it(tmp_path):
    reads = _crowd(tmp_path / "pos")
    other = _crowd(tmp_path / "neg")

    assert any_resolves_to(reads, {os.path.realpath(other[3])}) is False


def test_a_link_to_the_written_file_in_a_crowded_folder_is_found(tmp_path, links):
    reads = _crowd(tmp_path / "pos")
    written = tmp_path / "out" / "movie_data.csv"
    written.parent.mkdir()
    written.write_text("review,sentiment\n", encoding="utf-8")
    alias = tmp_path / "pos" / "alias.txt"
    os.symlink(written, alias)

    assert any_resolves_to([*reads, str(alias)], {os.path.realpath(written)}) is True


def test_a_document_read_through_a_linked_folder_is_found(tmp_path, links):
    reads = _crowd(tmp_path / "real")
    os.symlink(tmp_path / "real", tmp_path / "via", target_is_directory=True)
    through_link = [str(tmp_path / "via" / os.path.basename(p)) for p in reads]

    assert any_resolves_to(through_link, {os.path.realpath(reads[5])}) is True


def test_a_relative_spelling_is_resolved_from_the_working_folder(tmp_path, monkeypatch):
    _crowd(tmp_path / "aclImdb" / "pos")
    monkeypatch.chdir(tmp_path)
    reads = [os.path.join("aclImdb", "pos", f"{i}_7.txt") for i in range(N)]

    assert any_resolves_to(reads, {os.path.realpath(tmp_path / "aclImdb" / "pos" / "9_7.txt")}) is True
    assert any_resolves_to(reads, {os.path.realpath(tmp_path / "movie_data.csv")}) is False


def test_missing_files_answer_as_resolving_them_does(tmp_path):
    reads = [str(tmp_path / "gone" / f"{i}.txt") for i in range(N)]

    for target in (reads[2], str(tmp_path / "gone" / "other.txt")):
        targets = {os.path.realpath(target)}
        assert any_resolves_to(reads, targets) is _resolving_each(reads, targets)


def test_a_tree_of_files_and_links_answers_as_resolving_each(tmp_path, links):
    """Every file of a tree with linked files and a linked folder, against
    each of its files as the written one."""
    rng = random.Random(7)
    files = _crowd(tmp_path / "a") + _crowd(tmp_path / "b", 20)
    os.symlink(tmp_path / "b", tmp_path / "a" / "sub", target_is_directory=True)
    for i in range(10):
        os.symlink(rng.choice(files), tmp_path / "a" / f"link{i}.txt")
    reads = [str(tmp_path / "a" / name) for name in os.listdir(tmp_path / "a")]
    reads += [str(tmp_path / "a" / "sub" / name) for name in os.listdir(tmp_path / "b")]

    for target in [*files, str(tmp_path / "a" / "link3.txt"), str(tmp_path / "nothing.csv")]:
        targets = {os.path.realpath(target)}
        assert any_resolves_to(reads, targets) is _resolving_each(reads, targets), target
