"""A helper whose write cash's static check does not name runs every time.

``np.savetxt``, ``np.savez``, ``joblib.dump``, ``gzip.open(p, 'wt')``,
``zipfile.ZipFile(p, 'w')`` and ``os.open`` in a function of the user's are
none of the calls the static writer check reads, and ``p.touch()`` /
``os.remove(p)`` were read but not counted. The statement calling such a
helper was stored, and the next Run All restored its value without the
write: a folder cleared at the top stayed empty, and a staging file written
twice was read with its later content. What the statement is seen writing as
it runs now keeps it from being stored, and the static check counts every
call that changes a file by name.
"""

from __future__ import annotations

import os

import pytest

from cash.analysis.namespace_effects import user_callee_writing_files
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")


def _cells(body: str, path, counter, *, def_line: str = "") -> list[str]:
    """A helper counting its runs in *counter* (``os.open`` with
    ``O_APPEND``, which only the static check would count, and it is not in
    the user's ``open``) that writes *path* with *body*."""
    return [
        "import gzip, os, pathlib, tempfile, time, zipfile\nimport numpy as np\n"
        f"def export(arr, p):{def_line}\n"
        f"    fd = os.open({str(counter)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x'); os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        f"    {body}\n"
        "    return float(arr.sum())",
        "x = np.arange(4.0)",
        f"s = export(x, {str(path)!r})",
    ]


def _run_all(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


WRITES = {
    "savetxt": "np.savetxt(p, arr)",
    "savez": "np.savez(p, a=arr)",
    "gzip": "f = gzip.open(p, 'wt'); f.write(str(arr)); f.close()",
    "zipfile": "z = zipfile.ZipFile(p, 'w'); z.writestr('a', 'b'); z.close()",
    "os_open": "os.close(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC))",
    "touch": "pathlib.Path(p).touch()",
}


@pytest.mark.parametrize("body", list(WRITES.values()), ids=list(WRITES))
def test_the_file_is_written_again_on_every_run(cash_magics, tmp_path, body):
    out = tmp_path / "out.npz"  # np.savez adds the suffix to a name without it
    counter = tmp_path / "runs"
    cells = _cells(body, out, counter)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    _run_all(cash_magics, cells)
    assert out.exists()
    for _ in range(2):
        out.unlink()
        _run_all(cash_magics, cells)
        assert out.exists(), "a cache hit skipped the write"
    assert counter.read_bytes() == b"xxx"


def test_a_touch_of_a_file_that_is_there_is_a_write(cash_magics, tmp_path):
    """``touch`` on an existing file opens nothing: only the static check
    sees it."""
    out = tmp_path / "DONE"
    out.write_text("", encoding="utf-8")
    counter = tmp_path / "runs"
    cells = _cells("pathlib.Path(p).touch()", out, counter)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    _run_all(cash_magics, cells)
    _run_all(cash_magics, cells)
    assert counter.read_bytes() == b"xx"


def test_a_removal_in_a_helper_is_a_write(cash_magics, tmp_path):
    """``os.remove`` opens nothing either."""
    gone = tmp_path / "stale.csv"
    counter = tmp_path / "runs"
    cells = _cells("os.remove(p) if os.path.exists(p) else None", gone, counter)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    gone.write_text("old", encoding="utf-8")
    _run_all(cash_magics, cells)
    gone.write_text("old", encoding="utf-8")
    _run_all(cash_magics, cells)
    assert not gone.exists()


@pytest.mark.parametrize(
    "case",
    ["waived on the def", "temporary file", "only a counter"],
)
def test_what_a_hit_may_skip_is_still_served(cash_magics, tmp_path, case):
    """Controls: the user's waiver, a file removed again before the helper
    returns, and an append the user's code does not spell (the counter
    every case writes) leave the statement cacheable."""
    out = tmp_path / "out.txt"
    counter = tmp_path / "runs"
    body, def_line = {
        "waived on the def": ("np.savetxt(p, arr)", "  # @cash:assume-safe"),
        "temporary file": (
            "with tempfile.NamedTemporaryFile(suffix='.gz') as t: f = gzip.open(t, 'wt'); f.write('x'); f.close()",
            "",
        ),
        "only a counter": ("pass", ""),
    }[case]
    cells = _cells(body, out, counter, def_line=def_line)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        _run_all(cash_magics, cells)
    assert counter.read_bytes() == b"x"


def test_assume_safe_on_the_statement_caches_it(cash_magics, tmp_path):
    out = tmp_path / "out.txt"
    counter = tmp_path / "runs"
    cells = _cells("np.savetxt(p, arr)", out, counter)
    cells[2] += "  # @cash:assume-safe"
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        _run_all(cash_magics, cells)
    assert counter.read_bytes() == b"x"


def mark(p):
    open(p, "a", encoding="utf-8").close()
    os.utime(p)


def touch(p):
    p.touch()


def remove(p):
    os.remove(p)


def move(a, b):
    import shutil

    shutil.move(a, b)


def make_folder(p):
    os.makedirs(p, exist_ok=True)


def to_buffer(buf, text):
    buf.write(text)


def dump_into(fh, obj):
    import json

    json.dump(obj, fh)


def read_with(p, mode="r"):
    with open(p, mode) as fh:
        return fh.read()


@pytest.mark.parametrize("func", [touch, remove, move, make_folder])
def test_a_call_that_changes_a_file_by_name_is_a_writer(func):
    assert user_callee_writing_files(func) == func.__name__
    assert user_callee_writing_files(func, appends=False) == func.__name__


@pytest.mark.parametrize("func", [to_buffer, dump_into, read_with])
def test_a_write_on_a_handed_handle_or_a_computed_mode_is_not(func):
    """The handle may be a buffer in memory, and a computed mode as often
    reads: what the call is seen doing as it runs decides those."""
    assert user_callee_writing_files(func) is None


def test_an_append_alone_is_not_a_replacing_writer():
    assert user_callee_writing_files(mark) == "mark"
    assert user_callee_writing_files(mark, appends=False) is None
