"""The opt-in native file hook sees files that C libraries open themselves.

Python's audit hook reports every open made through Python; a C library that
opens its file itself raised nothing, and only the entry points
`reader_patches` wraps by name were tracked. With the hook on
(``CASH_NATIVE_FILE_HOOK=1``, `native_reads.enable`) the open family is
rebound in the loaded libraries' symbol tables, so a cached result depends on
those files too. Each case below shows today's behaviour with the hook off (a
stale answer) and the fix with it on:

- a small C library of the test's own that reads with ``fopen``/``open``
  (and on Windows with ``CreateFileW`` and ``_wfopen_s``);
- ``pyarrow.fs.LocalFileSystem().open_input_file`` (C++, not a wrapped reader);
- a database ``ATTACH``-ed in SQLite (the ``connect`` wrapper sees only the
  main file).

Plus the safety properties: errno and return values pass through, Python's own
opens are not recorded twice, threads and many reads are safe, a full buffer
makes the read unverifiable rather than lost, and disabling puts every slot back.
On Windows the hook patches import address tables instead of the GOT; the same
tests run there against a DLL built with MSVC.
"""

from __future__ import annotations

import contextlib
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

WINDOWS = sys.platform == "win32"

#: The test library's readers: the C library's on every system, the kernel's
#: and the wide CRT's on Windows.
READERS = ["sum_fopen", "sum_open"] + (["sum_createfile", "sum_wfopen"] if WINDOWS else [])

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not native_reads.supported()[0], reason=native_reads.supported()[1]),
]

C_SOURCE = r"""
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>
#define EXPORT __declspec(dllexport)
#define open _open
#define read _read
#define close _close
typedef int ssize_t;
#else
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#define EXPORT
#endif

static long sum_text(char *buf, long n) {
    long total = 0, v = 0; int in = 0;
    for (long i = 0; i <= n; i++) {
        if (buf[i] >= '0' && buf[i] <= '9') { v = v * 10 + (buf[i] - '0'); in = 1; }
        else if (in) { total += v; v = 0; in = 0; }
    }
    return total;
}

/* The sum of the integers in the file, or -errno. */
EXPORT long sum_fopen(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -errno;
    long total = 0, v;
    while (fscanf(f, "%ld", &v) == 1) total += v;
    fclose(f);
    return total;
}

EXPORT long sum_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -errno;
    char buf[4096];
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n < 0) return -1000;
    buf[n] = 0;
    return sum_text(buf, (long)n);
}

/* errno right after a failed open, as the caller sees it. */
EXPORT int open_errno(const char *path) {
    errno = 0;
    int fd = open(path, O_RDONLY);
    if (fd >= 0) { close(fd); return 0; }
    return errno;
}

/* Write *text* to *path* (creating or truncating it): 0 or -errno. */
EXPORT int write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f) return -errno;
    fputs(text, f);
    fclose(f);
    return 0;
}

/* Open *n* files named prefix0, prefix1, ... that do not exist. */
EXPORT long probe_many(const char *prefix, long n) {
    char name[4096];
    long missing = 0;
    for (long i = 0; i < n; i++) {
        snprintf(name, sizeof name, "%s%ld", prefix, i);
        int fd = open(name, O_RDONLY);
        if (fd < 0) missing++; else close(fd);
    }
    return missing;
}

#ifdef _WIN32
static wchar_t *wide(const char *utf8) {
    static __declspec(thread) wchar_t w[4096];
    if (!MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w, 4096)) return NULL;
    return w;
}

/* The same through the kernel: CreateFileW and ReadFile, or -GetLastError(). */
EXPORT long sum_createfile(const char *utf8) {
    HANDLE h = CreateFileW(wide(utf8), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return -(long)GetLastError();
    char buf[4096];
    DWORD n = 0;
    BOOL ok = ReadFile(h, buf, sizeof buf - 1, &n, NULL);
    CloseHandle(h);
    if (!ok) return -1000;
    buf[n] = 0;
    return sum_text(buf, (long)n);
}

/* And through the wide CRT: _wfopen_s. */
EXPORT long sum_wfopen(const char *utf8) {
    FILE *f = NULL;
    errno_t e = _wfopen_s(&f, wide(utf8), L"r");
    if (e) return -(long)e;
    long total = 0, v;
    while (fscanf(f, "%ld", &v) == 1) total += v;
    fclose(f);
    return total;
}

/* GetLastError right after a failed CreateFileW, as the caller sees it. */
EXPORT unsigned long createfile_error(const char *utf8) {
    SetLastError(0);
    HANDLE h = CreateFileW(wide(utf8), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); return 0; }
    return GetLastError();
}

/* Write through CreateFileW with *disposition*: 0 or -GetLastError(). */
EXPORT long createfile_write(const char *utf8, unsigned long disposition, const char *text) {
    HANDLE h = CreateFileW(wide(utf8), GENERIC_WRITE, 0, NULL, disposition, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return -(long)GetLastError();
    DWORD n;
    SetFilePointer(h, 0, NULL, FILE_END);
    WriteFile(h, text, (DWORD)strlen(text), &n, NULL);
    CloseHandle(h);
    return 0;
}

/* Nanoseconds per _open of *path* over *n* calls through the import table
 * ("" fails in the kernel at once). */
EXPORT double ns_per_open(const char *path, long n) {
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    for (long i = 0; i < n; i++) {
        int fd = open(path, O_RDONLY);
        if (fd >= 0) close(fd);
    }
    QueryPerformanceCounter(&t1);
    return (double)(t1.QuadPart - t0.QuadPart) * 1e9 / (double)f.QuadPart / (double)n;
}
#endif
"""


@pytest.fixture(scope="module")
def clib(tmp_path_factory):
    """The test's C library, built here and loaded with ctypes (after the
    hook was first enabled in some tests: a library loaded later is hooked on
    the next tracked block). On Windows it is a DLL built with MSVC against
    the shared CRT, so its ``fopen``/``_open``/``CreateFileW`` are real
    import-table entries."""
    folder = tmp_path_factory.mktemp("clib")
    src = folder / "reader.c"
    src.write_text(C_SOURCE, encoding="utf-8")
    if WINDOWS:
        lib = _build_dll_with_msvc(src, folder)
    else:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            pytest.skip("no C compiler")
        lib = folder / "libreader.so"
        subprocess.run([cc, "-shared", "-fPIC", "-O1", "-o", str(lib), str(src)], check=True)
    dll = ctypes.CDLL(str(lib), use_errno=True)
    for name in ("sum_fopen", "sum_open", "probe_many"):
        getattr(dll, name).restype = ctypes.c_long
    dll.probe_many.argtypes = (ctypes.c_char_p, ctypes.c_long)
    dll.open_errno.restype = ctypes.c_int
    dll.write_file.restype = ctypes.c_int
    if WINDOWS:
        for name in ("sum_createfile", "sum_wfopen", "createfile_write"):
            getattr(dll, name).restype = ctypes.c_long
        dll.createfile_error.restype = ctypes.c_ulong
        dll.createfile_write.argtypes = (ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p)
        dll.ns_per_open.restype = ctypes.c_double
        dll.ns_per_open.argtypes = (ctypes.c_char_p, ctypes.c_long)
    return dll


def _build_dll_with_msvc(src, folder):
    """*src* compiled to ``reader.dll`` by the compiler setuptools finds (MSVC)."""
    pytest.importorskip("setuptools")
    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.errors import DistutilsError

    compiler = new_compiler()
    try:
        with contextlib.chdir(folder):  # a relative source keeps the object next to it
            objects = compiler.compile([src.name])
            compiler.link_shared_object(objects, "reader.dll")
    except (DistutilsError, OSError) as exc:
        pytest.skip(f"no C compiler: {exc}")
    return folder / "reader.dll"


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


@pytest.mark.parametrize("reader", READERS)
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


@pytest.mark.parametrize("reader", READERS)
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
    if "_sqlite3" in sys.builtin_module_names:
        pytest.skip("sqlite is linked into this Python's executable, which the hook leaves alone")
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
        # A directory opens O_RDONLY on Linux; the CRT refuses it with EACCES.
        assert clib.open_errno(_b(tmp_path)) == (13 if WINDOWS else 0)
        if not WINDOWS:
            locked = tmp_path / "locked.txt"
            locked.write_text("1", encoding="utf-8")
            locked.chmod(0)
            if os.geteuid() != 0:
                assert clib.open_errno(_b(locked)) == 13  # EACCES
        assert clib.sum_fopen(_b(tmp_path / "nope")) == -2
        if WINDOWS:
            assert clib.createfile_error(_b(tmp_path / "nope")) == 2  # ERROR_FILE_NOT_FOUND
            assert clib.createfile_error(_b(tmp_path / "no" / "pe")) == 3  # ERROR_PATH_NOT_FOUND
            assert clib.sum_createfile(_b(tmp_path / "nope")) == -2
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


# On Windows the tracker takes ~20 s to record 65k missing paths (realpath is
# slow there); the hook's own part is a fraction of a second.
@pytest.mark.timeout(180)
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


# -- Windows: import address tables ----------------------------------------------

windows_only = pytest.mark.skipif(not WINDOWS, reason="the import-table back end is Windows only")


def _window_events(fn):
    window = native_reads.open_window()
    try:
        fn()
    finally:
        events, lost = native_reads._fh.filehook_window_close(window)
    assert not lost
    return {path: kind for kind, path in events}


@windows_only
def test_the_hooked_modules_are_the_test_library_not_the_system_or_python(clib, hook):
    native_reads._fh.filehook_rescan()
    hooked = [os.path.basename(p).lower() for p in native_reads.stats()["hooked_objects"]]
    assert "reader.dll" in hooked
    windir = os.environ.get("SystemRoot", r"C:\Windows").lower()
    for path in native_reads.stats()["hooked_objects"]:
        low = path.lower()
        assert not low.startswith(windir + "\\"), f"a system DLL was hooked: {path}"
        assert not os.path.basename(low).startswith(("python3", "ucrtbase", "vcruntime")), path


@windows_only
def test_createfile_dispositions_are_classified(tmp_path, clib, hook):
    """CREATE_ALWAYS makes the file (the block's output); OPEN_ALWAYS on a
    file that was there writes to it; OPEN_ALWAYS on a missing one makes it."""
    CREATE_ALWAYS, OPEN_ALWAYS = 2, 4
    made, kept, fresh = tmp_path / "made.txt", tmp_path / "kept.txt", tmp_path / "fresh.txt"
    kept.write_text("x", encoding="utf-8")

    def run():
        assert clib.createfile_write(_b(made), CREATE_ALWAYS, b"1") == 0
        assert clib.createfile_write(_b(kept), OPEN_ALWAYS, b"2") == 0
        assert clib.createfile_write(_b(fresh), OPEN_ALWAYS, b"3") == 0

    kinds = _window_events(run)
    assert kinds[_b(made)] == native_reads._CREATED
    assert kinds[_b(kept)] == native_reads._WRITE
    assert kinds[_b(fresh)] == native_reads._CREATED


@windows_only
def test_a_folder_is_not_a_read(tmp_path, clib, hook):
    kinds = _window_events(lambda: clib.sum_fopen(_b(tmp_path)))
    assert _b(tmp_path) not in kinds


@windows_only
def test_last_error_passes_through_a_hooked_call(tmp_path, clib, hook):
    """The hook's own work (GetFullPathNameW, a stat) must not change what
    GetLastError says after the real call."""
    present = tmp_path / "here.txt"
    present.write_text("1", encoding="utf-8")
    with FileAccessTracker():
        assert clib.createfile_error(_b(tmp_path / "missing")) == 2
        assert clib.createfile_error(_b(present)) == 0


@windows_only
def test_calls_through_getprocaddress_are_not_seen(tmp_path, hook):
    """The documented limit: ctypes (like any GetProcAddress caller) calls the
    CRT's function directly, not through an import table."""
    path = tmp_path / "direct.txt"
    path.write_text("1", encoding="utf-8")
    crt = ctypes.CDLL("ucrtbase")
    crt.fopen.restype = ctypes.c_void_p
    crt.fopen.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
    crt.fclose.argtypes = (ctypes.c_void_p,)

    def run():
        f = crt.fopen(_b(path), b"r")
        assert f
        crt.fclose(f)

    assert _b(path) not in _window_events(run), "the gap closed: update this test and the docs"


@windows_only
def test_a_dll_loaded_after_enable_is_hooked_at_the_next_window(tmp_path, clib, hook):
    """New modules are noticed (a loader notification) and patched when the
    next tracker opens; reader.dll was loaded after the first enable."""
    path = tmp_path / "late.txt"
    path.write_text("2 2", encoding="utf-8")
    native_reads.disable()
    assert native_reads.enable()
    assert _b(path) in _window_events(lambda: clib.sum_createfile(_b(path)))


@windows_only
@pytest.mark.perf
def test_an_unrecorded_open_costs_well_under_a_microsecond_more(clib):
    """With the hook on and no tracker open, a hooked call is one load and a
    jump. Timed inside the DLL (200k ``_open("")`` calls, which fail in the
    kernel at once), best of five, hook off against hook on."""

    def per_call():
        return min(clib.ns_per_open(b"", 200_000) for _ in range(5))

    native_reads.disable()
    off = per_call()
    assert native_reads.enable()
    try:
        native_reads._fh.filehook_rescan()
        on = per_call()
    finally:
        native_reads.disable()
    assert on - off < 1000, f"hook on adds {on - off:.0f} ns per open ({off:.0f} -> {on:.0f} ns)"
