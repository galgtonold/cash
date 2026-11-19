"""A file a cached function reads through ``os.open`` is an input, as one read
through ``open`` is.

``fd = os.open(p, os.O_RDONLY); os.read(fd, n)`` raises the same ``open``
audit event as ``open(p, "rb")``, with the flags in place of a mode, and the
native file hook leaves the interpreter's own calls to that event. The
tracker took that event for a write only, so the read was never an input:
after the file changed, the function kept returning the old bytes, with no
warning.
"""

from __future__ import annotations

import os

import pytest


def _settle(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _write(path, data: bytes):
    path.write_bytes(data)
    _settle(path)


def _read_with(flags):
    def read(p):
        fd = os.open(p, flags)
        try:
            return os.read(fd, 100)
        finally:
            os.close(fd)

    return read


@pytest.mark.parametrize(
    "flags",
    [os.O_RDONLY, os.O_RDWR, os.O_RDONLY | getattr(os, "O_BINARY", 0)],
    ids=["read_only", "read_write", "binary"],
)
def test_an_edit_to_the_file_recomputes(disk_cash, tmp_path, flags):
    path = tmp_path / "data.bin"
    read = _read_with(flags)
    runs: list[int] = []

    @disk_cash.cache
    def load(p):
        runs.append(1)
        return read(p)

    _write(path, b"one")
    assert load(str(path)) == b"one"
    assert load(str(path)) == b"one"
    assert len(runs) == 1  # positive control: an unchanged file hits
    _write(path, b"twotwo")
    assert load(str(path)) == b"twotwo"
    assert len(runs) == 2


def test_the_file_is_listed_as_a_dependency(disk_cash, tmp_path):
    path = tmp_path / "data.bin"
    _write(path, b"one")
    read = _read_with(os.O_RDONLY)

    @disk_cash.cache
    def load(p):
        return read(p)

    load(str(path))
    assert str(path.resolve()) in str(load.explain(str(path)))


def test_a_file_that_appears_recomputes(disk_cash, tmp_path):
    path = tmp_path / "late.bin"
    read = _read_with(os.O_RDONLY)

    @disk_cash.cache
    def load(p):
        try:
            return read(p)
        except FileNotFoundError:
            return b"MISSING"

    assert load(str(path)) == b"MISSING"
    _write(path, b"here")
    assert load(str(path)) == b"here"


def test_a_file_opened_only_to_append_is_no_input(disk_cash, tmp_path):
    # The suite counts runs with this append (`.github/copilot-instructions.md`):
    # it must stay a write, not become an input that changes on every run.
    log = tmp_path / "runs.log"
    data = tmp_path / "data.bin"
    _write(data, b"one")
    read = _read_with(os.O_RDONLY)

    @disk_cash.cache(assume_safe=True)
    def load(p):
        fd = os.open(str(log), os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(fd, b"a\n")
        os.close(fd)
        return read(p)

    load(str(data))
    load(str(data))
    assert log.read_text(encoding="utf-8").split() == ["a"]
