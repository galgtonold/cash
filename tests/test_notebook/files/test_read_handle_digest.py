"""A file open for reading is named by its file and where it stands.

Two handles opened on the same unchanged file at the same place get the same
digest, so what is read from one restores on a later run; a read, an edit or
a write handle tells them apart.
"""

import os

from cash.notebook.lineage_formula import no_cache_value_digest
from cash.notebook.read_handles import read_handle_digest


def test_two_opens_of_the_same_file_name_the_same_value(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    with open(path, encoding="utf-8") as one, open(path, encoding="utf-8") as two:
        assert read_handle_digest(one) == read_handle_digest(two)
        assert no_cache_value_digest(one) == read_handle_digest(one)


def test_a_read_moves_the_digest(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        before = read_handle_digest(fh)
        fh.readline()
        assert read_handle_digest(fh) != before


def test_an_edited_file_names_a_new_value(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        before = read_handle_digest(fh)
    path.write_text("a\nc\nd\n", encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        assert read_handle_digest(fh) != before


def test_decoding_is_part_of_the_value(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a\n", encoding="utf-8")
    with open(path, encoding="utf-8") as text, open(path, encoding="latin-1") as other, open(path, "rb") as raw:
        assert len({read_handle_digest(text), read_handle_digest(other), read_handle_digest(raw)}) == 3


def test_other_streams_keep_a_fresh_value(tmp_path):
    path = tmp_path / "out.txt"
    with open(path, "w", encoding="utf-8") as out:
        assert read_handle_digest(out) is None
        assert no_cache_value_digest(out).startswith("fresh:")
    with open(path, "r+", encoding="utf-8") as both:
        assert read_handle_digest(both) is None
    fh = open(path, encoding="utf-8")
    fh.close()
    assert read_handle_digest(fh) is None
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as pipe, os.fdopen(write_fd, "w"):
        assert read_handle_digest(pipe) is None
