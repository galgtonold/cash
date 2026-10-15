"""The opt-in native file hook sees files that C libraries open themselves.

Python's audit hook reports every open made through Python; a C library that
opens its file itself raised nothing, and only the entry points
`reader_patches` wraps by name were tracked. With the hook on
(``CASH_NATIVE_FILE_HOOK=1``, `native_reads.enable`) the open family is
rebound in the loaded libraries' symbol tables, so a cached result depends on
those files too. Each case below shows today's behaviour with the hook off (a
stale answer) and the fix with it on:

- a small C library of the test's own that reads with ``fopen``/``open``;
- ``pyarrow.fs.LocalFileSystem().open_input_file`` (C++, not a wrapped reader);
- a database ``ATTACH``-ed in SQLite (the ``connect`` wrapper sees only the
  main file).

Plus the safety properties: errno and return values pass through, Python's own
opens are not recorded twice, threads and many reads are safe, a full buffer
makes the read unverifiable rather than lost, and disabling puts every slot back.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sqlite3
import subprocess
import sys
import threading

import pytest

from cash.tracking import native_reads
from cash.tracking.file_tracker import FileAccessTracker

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not native_reads.supported()[0], reason=native_reads.supported()[1]),
]

C_SOURCE = r"""
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

/* The sum of the integers in the file, or -errno. */
long sum_fopen(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -errno;
    long total = 0, v;
    while (fscanf(f, "%ld", &v) == 1) total += v;
    fclose(f);
    return total;
}

long sum_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -errno;
    char buf[4096];
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n < 0) return -1000;
    buf[n] = 0;
    long total = 0, v = 0; int in = 0;
    for (ssize_t i = 0; i <= n; i++) {
        if (buf[i] >= '0' && buf[i] <= '9') { v = v * 10 + (buf[i] - '0'); in = 1; }
        else if (in) { total += v; v = 0; in = 0; }
    }
    return total;
}

/* errno right after a failed open, as the caller sees it. */
int open_errno(const char *path) {
    errno = 0;
    int fd = open(path, O_RDONLY);
    if (fd >= 0) { close(fd); return 0; }
    return errno;
}

/* Write *text* to *path* (creating or truncating it): 0 or -errno. */
int write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f) return -errno;
    fputs(text, f);
    fclose(f);
    return 0;
}

/* Open *n* files named prefix0, prefix1, ... that do not exist. */
long probe_many(const char *prefix, long n) {
    char name[4096];
    long missing = 0;
    for (long i = 0; i < n; i++) {
        snprintf(name, sizeof name, "%s%ld", prefix, i);
        int fd = open(name, O_RDONLY);
        if (fd < 0) missing++; else close(fd);
    }
    return missing;
}
"""


@pytest.fixture(scope="module")
def clib(tmp_path_factory):
    """The test's C library, built here and loaded with ctypes (after the
    hook was first enabled in some tests: a library loaded later is hooked on
    the next tracked block)."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    folder = tmp_path_factory.mktemp("clib")
    src = folder / "reader.c"
    src.write_text(C_SOURCE, encoding="utf-8")
    lib = folder / "libreader.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O1", "-o", str(lib), str(src)], check=True)
    dll = ctypes.CDLL(str(lib), use_errno=True)
    for name in ("sum_fopen", "sum_open", "probe_many"):
        getattr(dll, name).restype = ctypes.c_long
    dll.probe_many.argtypes = (ctypes.c_char_p, ctypes.c_long)
    dll.open_errno.restype = ctypes.c_int
    dll.write_file.restype = ctypes.c_int
    return dll


@pytest.fixture
def hook():
    """The hook on for this test, then back as it was."""
    was = native_reads.active()
    assert native_reads.enable()
    yield
    if not was:
        native_reads.disable()


@pytest.fixture
def no_hook():
    was = native_reads.active()
    native_reads.disable()
    yield
    if was:
        native_reads.enable()


def _b(path) -> bytes:
    return os.fsencode(str(path))


def _cached_sum(disk_cash, read):
    # The ctypes library is keyed by its file, like any loaded code.
    disk_cash.register_hasher(ctypes.CDLL, lambda lib: lib._name, override=True)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        return read(p)

    return total, runs


# -- what the hook catches that the reader patches miss ---------------------------


@pytest.mark.parametrize("reader", ["sum_fopen", "sum_open"])
def test_a_c_library_read_is_stale_without_the_hook(disk_cash, tmp_path, clib, no_hook, reader):
    """Today's behaviour, pinned so the gap stays documented: a file read by a
    C library is no dependency, so its new content is served the old answer."""
    path = tmp_path / "data.txt"
    path.write_text("1 2 3\n", encoding="utf-8")
    total, runs = _cached_sum(disk_cash, lambda p: getattr(clib, reader)(_b(p)))
    assert total(str(path)) == 6
    path.write_text("10 20 30 40\n", encoding="utf-8")
    assert total(str(path)) == 6, "the gap closed without the hook: update this test and the docs"
    assert len(runs) == 1


@pytest.mark.parametrize("reader", ["sum_fopen", "sum_open"])
def test_a_c_library_read_is_a_dependency_with_the_hook(disk_cash, tmp_path, clib, hook, reader):
    path = tmp_path / "data.txt"
    path.write_text("1 2 3\n", encoding="utf-8")
    total, runs = _cached_sum(disk_cash, lambda p: getattr(clib, reader)(_b(p)))
    assert total(str(path)) == 6
    assert total(str(path)) == 6
    assert len(runs) == 1, "an unchanged file recomputed"
    path.write_text("10 20 30 40\n", encoding="utf-8")
    assert total(str(path)) == 100, "the C library's file changed and the old answer was served"
    assert len(runs) == 2


def test_a_missing_file_a_c_library_looked_for_is_an_absent_dependency(disk_cash, tmp_path, clib, hook):
    path = tmp_path / "optional.txt"
    total, runs = _cached_sum(disk_cash, lambda p: max(clib.sum_fopen(_b(p)), 0))
    assert total(str(path)) == 0
    assert total(str(path)) == 0
    assert len(runs) == 1
    path.write_text("5 6\n", encoding="utf-8")
    assert total(str(path)) == 11, "a file that appeared was not noticed"


def test_a_file_a_c_library_writes_and_reads_back_is_its_own_output(disk_cash, tmp_path, clib, hook):
    """A scratch file the block writes, reads and removes is not an input
    (otherwise the entry would depend on a file that no longer exists)."""
    disk_cash.register_hasher(ctypes.CDLL, lambda lib: lib._name, override=True)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def scratch(n):
        runs.append(1)
        p = tmp_path / f"scratch{n}.txt"
        assert clib.write_file(_b(p), b"4 5\n") == 0
        value = clib.sum_fopen(_b(p))
        os.remove(p)
        return value

    assert scratch(1) == 9
    assert scratch(1) == 9
    assert len(runs) == 1, "a file the block made itself became an input it can never find again"


def test_pyarrow_open_input_file_is_seen(disk_cash, tmp_path, request):
    """``LocalFileSystem.open_input_file`` is no wrapped reader: stale without
    the hook, a dependency with it."""
    pafs = pytest.importorskip("pyarrow.fs")

    def read(p):
        with pafs.LocalFileSystem().open_input_file(p) as f:
            return f.read().decode()

    for on in (False, True):
        (native_reads.enable if on else native_reads.disable)()
        request.addfinalizer(native_reads.disable)
        path = tmp_path / f"blob{on}.bin"
        path.write_text(f"first {on}", encoding="utf-8")
        total, runs = _cached_sum(disk_cash, read)
        assert total(str(path)) == f"first {on}"
        path.write_text(f"second {on}!", encoding="utf-8")
        got = total(str(path))
        if on:
            assert got == f"second {on}!", "pyarrow's own open was not seen with the hook on"
        else:
            assert got == f"first {on}", "the gap closed without the hook: update this test and the docs"


def test_an_attached_sqlite_database_is_seen(disk_cash, tmp_path, request):
    """The ``connect`` wrapper records the main database; one ``ATTACH``-ed
    in SQL is opened by libsqlite3 itself."""
    main, other = tmp_path / "main.db", tmp_path / "other.db"
    with sqlite3.connect(main) as conn:
        conn.execute("create table if not exists m(x)")
    conn.close()

    def set_rows(values):
        with sqlite3.connect(other) as conn:
            conn.execute("create table if not exists t(x)")
            conn.execute("delete from t")
            conn.executemany("insert into t values (?)", [(v,) for v in values])
        conn.close()

    def read(_):
        conn = sqlite3.connect(main)
        try:
            conn.execute("attach database ? as o", (str(other),))
            return conn.execute("select coalesce(sum(x), 0) from o.t").fetchone()[0]
        finally:
            conn.close()

    for on in (False, True):
        (native_reads.enable if on else native_reads.disable)()
        request.addfinalizer(native_reads.disable)
        set_rows([1, 2, 3])
        total, runs = _cached_sum(disk_cash, read)
        assert total(f"{on}") == 6
        set_rows([10, 20])
        got = total(f"{on}")
        assert got == (30 if on else 6), f"hook {'on' if on else 'off'}: got {got}"


# -- what a tracker gets, and what it does not -------------------------------------


def _reads_of(fn):
    with FileAccessTracker() as tracker:
        fn()
    return tracker


def test_python_opens_are_not_recorded_twice(tmp_path, hook):
    path = tmp_path / "py.txt"
    path.write_text("x", encoding="utf-8")
    window = native_reads.open_window()
    with open(path, encoding="utf-8") as f:
        f.read()
    events, lost = native_reads._fh.filehook_window_close(window)
    assert [e for e in events if e[1] == _b(path)] == [], "Python's own open came through the native hook"
    assert not lost


def test_errno_and_results_pass_through(tmp_path, clib, hook):
    window = native_reads.open_window()
    try:
        assert clib.open_errno(_b(tmp_path / "nope")) == 2  # ENOENT
        assert clib.open_errno(_b(tmp_path)) == 0  # a directory opens O_RDONLY
        locked = tmp_path / "locked.txt"
        locked.write_text("1", encoding="utf-8")
        locked.chmod(0)
        if os.geteuid() != 0:
            assert clib.open_errno(_b(locked)) == 13  # EACCES
        assert clib.sum_fopen(_b(tmp_path / "nope")) == -2
    finally:
        events, _ = native_reads._fh.filehook_window_close(window)
    kinds = {path: kind for kind, path in events}
    assert kinds[_b(tmp_path / "nope")] == native_reads._ABSENT
    assert _b(tmp_path) not in kinds, "a directory opened as a descriptor is not a read of it"


def test_relative_paths_are_made_absolute_when_opened(tmp_path, clib, hook, monkeypatch):
    (tmp_path / "rel.txt").write_text("7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    tracker = _reads_of(lambda: clib.sum_fopen(b"rel.txt"))
    assert os.path.realpath(tmp_path / "rel.txt") in {os.path.realpath(p) for p in tracker.get_accessed_files()}


def test_reads_from_many_threads_reach_the_tracker(tmp_path, clib, hook):
    paths = []
    for i in range(16):
        p = tmp_path / f"t{i}.txt"
        p.write_text(f"{i}\n", encoding="utf-8")
        paths.append(p)

    def work(chunk):
        for _ in range(50):
            for p in chunk:
                clib.sum_fopen(_b(p))

    def run():
        threads = [threading.Thread(target=work, args=(paths[i::4],)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    tracker = _reads_of(run)
    seen = {os.path.realpath(p) for p in tracker.get_accessed_files()}
    assert {os.path.realpath(p) for p in paths} <= seen


def test_nested_trackers_each_get_the_reads_made_while_open(tmp_path, clib, hook):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("1", encoding="utf-8")
    b.write_text("2", encoding="utf-8")
    with FileAccessTracker() as outer:
        clib.sum_fopen(_b(a))
        with FileAccessTracker() as inner:
            clib.sum_fopen(_b(b))
    real = os.path.realpath
    assert {real(a), real(b)} <= {real(p) for p in outer.get_accessed_files()}
    assert real(b) in {real(p) for p in inner.get_accessed_files()}
    assert real(a) not in {real(p) for p in inner.get_accessed_files()}
    assert native_reads.stats()["windows"] == 0
    assert native_reads.stats()["buffered"] == 0, "events outlived every window"


def test_a_full_buffer_makes_the_block_unverifiable(tmp_path, clib, hook):
    """More distinct opens than the buffer holds: the tracker records a
    dependency that is never fresh instead of silently missing some."""
    tracker = _reads_of(lambda: clib.probe_many(_b(tmp_path / "missing-"), 70_000))
    assert native_reads.LOST_READS in tracker.unresolved_files


def test_no_recording_outside_a_tracker(tmp_path, clib, hook):
    before = native_reads.stats()["recorded"]
    clib.sum_fopen(_b(tmp_path / "x"))
    assert native_reads.stats()["recorded"] == before


def test_disable_puts_every_slot_back(tmp_path, clib):
    assert native_reads.enable()
    assert native_reads.stats()["slots"] > 0
    native_reads.disable()
    assert native_reads.stats()["slots"] == 0
    assert native_reads.open_window() is None
    # The library still works through the original functions.
    p = tmp_path / "after.txt"
    p.write_text("3 4", encoding="utf-8")
    assert clib.sum_fopen(_b(p)) == 7
    assert clib.sum_open(_b(p)) == 7
    # And again on: the same slots come back.
    assert native_reads.enable()
    assert native_reads.stats()["slots"] > 0
    native_reads.disable()


def test_status_line_says_on_or_off():
    native_reads.disable()
    assert native_reads.status_line().startswith("off (set CASH_NATIVE_FILE_HOOK=1")
    native_reads.enable()
    try:
        assert native_reads.status_line().startswith("on (")
    finally:
        native_reads.disable()


def test_the_environment_switch_turns_it_on_at_import(tmp_path):
    code = "from cash.tracking import native_reads as n; print(n.active(), n.stats()['slots'] > 0)"
    env = dict(os.environ, CASH_NATIVE_FILE_HOOK="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["True", "True"]
