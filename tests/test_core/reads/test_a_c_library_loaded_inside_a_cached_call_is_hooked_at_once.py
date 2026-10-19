"""A C library first loaded inside a cached call has its file reads seen.

The native file hook rebinds the open family in the libraries loaded when a
tracked block opens, so a library the block itself loaded -- ``import
pyarrow.fs`` in the body of the cached function, a ``ctypes.CDLL`` made there
-- opened its file unseen. The entry kept no file and a hit never runs the
body again, so the edited file's old result was served, in every later
process too, with no warning. An extension module's import and a
``ctypes.CDLL`` load inside a block now hook the new library before its code
runs.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from cash.tracking import native_reads
from tests.test_core.reads.test_the_native_file_hook_sees_c_library_reads import (
    C_SOURCE,
    WINDOWS,
    _build_dll_with_msvc,
)

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not native_reads.supported()[0], reason=native_reads.supported()[1]),
]


@pytest.fixture
def library(tmp_path):
    """The reader library built afresh, at a path no process has loaded yet."""
    folder = tmp_path / "lib"
    folder.mkdir()
    src = folder / "reader.c"
    src.write_text(C_SOURCE, encoding="utf-8")
    if WINDOWS:
        return str(_build_dll_with_msvc(src, folder))
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    lib = folder / "libreader_late.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O1", "-o", str(lib), str(src)], check=True)
    return str(lib)


@pytest.fixture
def hook():
    was = native_reads.active()
    assert native_reads.enable()
    yield
    if not was:
        native_reads.disable()


def test_a_library_loaded_with_ctypes_inside_the_call(disk_cash, tmp_path, library, hook):
    data = tmp_path / "data.txt"
    data.write_text("1 2 3\n", encoding="utf-8")
    runs = []

    @disk_cash.cache(assume_safe=True)
    def total(lib_path, path):
        runs.append(1)
        lib = ctypes.CDLL(lib_path)  # loaded here, inside the tracked block
        lib.sum_fopen.restype = ctypes.c_long
        return lib.sum_fopen(os.fsencode(path))

    assert total(library, str(data)) == 6
    assert total(library, str(data)) == 6
    assert len(runs) == 1, "an unchanged file recomputed"
    data.write_text("10 20 30 40\n", encoding="utf-8")
    assert total(library, str(data)) == 100, "the late-loaded library's file changed and the old answer came back"
    assert len(runs) == 2


_LATE_IMPORT = textwrap.dedent(
    """
    import sys, warnings
    import cash
    assert "pyarrow" not in sys.modules
    c = cash.Cash(cache_dir=sys.argv[1])

    @c.cache
    def head(path):
        import pyarrow.fs  # first loaded here, inside the tracked block
        with pyarrow.fs.LocalFileSystem().open_input_stream(path) as s:
            return s.read()

    warnings.simplefilter("ignore")
    print(head(sys.argv[2]).decode().strip())
    """
)


def test_an_extension_module_imported_inside_the_call(tmp_path):
    """In a process of its own, so pyarrow is not loaded yet."""
    pytest.importorskip("pyarrow.fs")
    script = tmp_path / "late_import.py"
    script.write_text(_LATE_IMPORT, encoding="utf-8")
    data = tmp_path / "data.txt"
    data.write_text("old\n", encoding="utf-8")

    def run():
        out = subprocess.run(
            [sys.executable, str(script), str(tmp_path / "cache"), str(data)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, native_reads.ENV: "1"},
        )
        return out.stdout.strip()

    assert run() == "old"
    data.write_text("new\n", encoding="utf-8")
    assert run() == "new", "the file pyarrow read changed and the old answer was served"


def test_a_load_outside_a_tracked_block_scans_nothing(library, hook):
    """Outside a block the next window hooks what was loaded, once for all."""
    before = native_reads.stats()["objects_scanned"]
    ctypes.CDLL(library)
    assert native_reads.stats()["objects_scanned"] == before
