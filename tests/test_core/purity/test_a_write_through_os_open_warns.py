"""A helper writing through ``os.open`` + ``os.write`` is a write, as
``open(p, 'a')`` in the same helper is: the first call warns.

``os.open`` raises the ``open`` audit event with no mode, and the file
tracker returned on it before the effect observer heard of the write. A
cached function whose helper appended a line through ``os.open`` -- one the
static pass does not read -- was stored without a word, and every hit
skipped the line.
"""

from __future__ import annotations

import importlib.util
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpurityWarning

FLAGS = {
    "append": "os.O_WRONLY | os.O_APPEND | os.O_CREAT",
    "truncate": "os.O_WRONLY | os.O_TRUNC | os.O_CREAT",
    "read-write-append": "os.O_RDWR | os.O_APPEND | os.O_CREAT",
}


def _helpers(tmp_path, body):
    """A module of helpers loaded from a file and kept out of
    ``sys.modules``, which the static pass does not read: only the observer
    can see what they do."""
    source = tmp_path / "raw_helpers.py"
    source.write_text("import os, tempfile\n\n" + body)
    spec = importlib.util.spec_from_file_location("raw_helpers", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _warnings(tmp_path, func, *args):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(func)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cached(*args)
        cached(*args)
    return [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]


@pytest.mark.parametrize("flags", list(FLAGS.values()), ids=list(FLAGS))
def test_a_helper_writing_through_os_open_warns(tmp_path, flags):
    log = tmp_path / "run.log"
    helpers = _helpers(
        tmp_path,
        f"def write_raw(path):\n    fd = os.open(path, {flags})\n    os.write(fd, b'a\\n')\n    os.close(fd)\n    return path\n",
    )

    def logged(x):
        written = helpers.write_raw(str(log))
        return x, written

    found = _warnings(tmp_path, logged, 1)
    assert len(found) == 1, found
    assert "IMPURE-OBSERVED-EFFECTS" in found[0] and "file write" in found[0] and "run.log" in found[0]
    assert log.read_bytes() == b"a\n"  # the hit skipped it, as warned


def test_a_read_through_os_open_is_no_write(tmp_path):
    data = tmp_path / "data.txt"
    data.write_bytes(b"x")
    helpers = _helpers(
        tmp_path, "def read_raw(path):\n    fd = os.open(path, os.O_RDONLY)\n    os.close(fd)\n    return path\n"
    )

    def reads(path):
        return helpers.read_raw(path)

    assert not [m for m in _warnings(tmp_path, reads, str(data)) if "OBSERVED" in m]


def test_a_temporary_file_is_no_effect(tmp_path):
    helpers = _helpers(
        tmp_path,
        "def scratch(folder):\n"
        "    fd, name = tempfile.mkstemp(dir=folder)\n"
        "    os.close(fd)\n"
        "    os.remove(name)\n"
        "    with tempfile.TemporaryFile(dir=folder) as f:\n"
        "        f.write(b'x')\n"
        "    with tempfile.NamedTemporaryFile(dir=folder) as f:\n"
        "        f.write(b'x')\n"
        "    return folder\n",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def uses_scratch(folder):
        return helpers.scratch(folder)

    assert not [m for m in _warnings(tmp_path, uses_scratch, str(scratch)) if "OBSERVED" in m]


def test_a_waived_os_open_write_is_silent(tmp_path):
    log = tmp_path / "runs.log"
    helpers = _helpers(
        tmp_path,
        "def count(path):\n"
        "    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.close(fd)\n"
        "    return path\n",
    )

    def counted(x):
        counter = helpers.count(str(log))
        return x, counter

    assert not [m for m in _warnings(tmp_path, counted, 1) if "OBSERVED" in m]
