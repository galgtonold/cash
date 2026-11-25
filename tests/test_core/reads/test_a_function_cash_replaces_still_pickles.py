"""A library function cash replaces still pickles, as the function it was.

While a cached call runs, cash puts a wrapper in the place of the readers
and probes that raise no audit event (``os.path.getsize``, ``os.stat``,
``sqlite3.connect``). Programs pass those functions to process pools --
``pool.map(os.path.getsize, files)`` -- and pickle stores a function by its
name and checks the name still gives that object. Cash's wrappers failed the
check, so a function that works plain raised ``PicklingError`` under
``@cash.cache``. A probe a pool runs as its task is an input of the call
like one the body makes itself.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import pickle
import sys
import types

import pytest

from cash.tracking.file_tracker import FileAccessTracker

pytestmark = pytest.mark.core


def _module_callables() -> dict[tuple[str, str], object]:
    """Every callable bound at the top of an imported library module."""
    found = {}
    for module_name, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType) or module_name.startswith(("cash", "tests", "test_", "conftest")):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            if callable(value) and not isinstance(value, type):
                found[module_name, name] = value
    return found


def test_every_function_replaced_while_a_call_is_tracked_pickles_as_the_original():
    """The hygiene check: whatever cash swaps into a library module while a
    tracker is open pickles, loads back as what stands under that name, and
    is the original again once the tracker has closed."""
    import numpy  # noqa: F401 - loaded so their readers are among the replaced
    import pandas  # noqa: F401 - loaded so their readers are among the replaced

    before = _module_callables()
    replaced = {}
    with FileAccessTracker():
        for where, now in _module_callables().items():
            original = before.get(where)
            if original is not None and now is not original:
                replaced[where] = (original, now, pickle.dumps(now))
                assert pickle.loads(replaced[where][2]) is now, where
                assert copy.deepcopy(now) is now and copy.copy(now) is now, where
    names = {f"{m}.{n}" for m, n in replaced}
    # Positive control: the scan saw the wrappers this is about.
    for expected in (
        "posixpath.getsize" if os.name == "posix" else "ntpath.getsize",
        "os.stat",
        "os.lstat",
        "os.access",
        "os.scandir",
        "posixpath.exists" if os.name == "posix" else "ntpath.exists",
        "posixpath.isfile" if os.name == "posix" else "ntpath.isfile",
        "sqlite3.connect",
        "linecache.getline",
        "pandas.read_csv",
    ):
        assert expected in names, f"{expected} was not replaced while a tracker was open: {sorted(names)[:40]}"
    for where, (original, _wrapper, data) in replaced.items():
        assert getattr(sys.modules[where[0]], where[1]) is original, where
        assert pickle.loads(data) is original, where


def test_one_original_has_one_replacement_wherever_it_is_exposed():
    import sqlite3
    import sqlite3.dbapi2

    with FileAccessTracker():
        assert sqlite3.connect is sqlite3.dbapi2.connect
        assert os.path.getsize is sys.modules[os.path.__name__].getsize


def test_only_the_names_cash_watches_are_replaced_not_where_the_original_says_it_lives():
    """``os.stat`` calls itself ``posix.stat``, and the import system and
    the standard library call it there: set there too, every ``import`` in
    a tracked block went through cash's wrapper, and a statement served
    after a restart was keyed apart from the run that stored it."""
    low_level = sys.modules[os.stat.__module__]
    originals = (low_level.stat, low_level.listdir, low_level.scandir, low_level.access)
    with FileAccessTracker():
        assert os.stat is not originals[0], "the watched name is replaced"
        assert (low_level.stat, low_level.listdir, low_level.scandir, low_level.access) == originals
        assert pickle.loads(pickle.dumps(os.stat)) is os.stat
    assert pickle.loads(pickle.dumps(os.stat)) is originals[0]


def test_a_probe_keys_the_same_inside_another_cached_call(disk_cash, tmp_path):
    """While a tracker is open ``os.stat`` is cash's wrapper, under another
    name than the original's own: a call nested in another cached call
    still finds the entry a top-level call stored."""
    import sqlite3

    path = tmp_path / "d.txt"
    path.write_text("abc", encoding="utf-8")
    runs: list[str] = []

    @disk_cash.cache(assume_safe=True)
    def inner(p):
        runs.append("inner")
        return os.stat(p).st_size + os.path.getsize(p) + int(os.access(p, os.R_OK)) + len(sqlite3.connect.__name__)

    @disk_cash.cache(assume_safe=True)
    def outer(p):
        runs.append("outer")
        return inner(p) + 1

    assert inner(str(path)) == 14
    assert outer(str(path)) == 15
    assert runs == ["inner", "outer"]


def _sizes(files):
    with mp.get_context("fork").Pool(2) as pool:
        return pool.map(os.path.getsize, files)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
def test_a_cached_function_maps_a_probe_over_a_process_pool(disk_cash, tmp_path, recwarn):
    files = []
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(name * 3, encoding="utf-8")
        files.append(str(tmp_path / name))
    files = tuple(files)
    expected = _sizes(files)
    cached = disk_cash.cache(_sizes)
    assert cached(files) == expected == [15, 15]
    # The workers' probes are still the call's inputs.
    (tmp_path / "a.txt").write_text("longer than before", encoding="utf-8")
    assert cached(files) == [18, 15]


def _probes_in_a_pool(files):
    with mp.get_context("fork").Pool(2) as pool:
        return (
            pool.map(os.path.exists, files),
            pool.map(os.path.isfile, files),
            [r.st_size for r in pool.map(os.stat, files)],
            pool.starmap(os.access, [(f, os.R_OK) for f in files]),
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
def test_every_probe_a_cached_call_replaces_goes_through_a_pool(disk_cash, tmp_path, recwarn):
    (tmp_path / "a.txt").write_text("abc", encoding="utf-8")
    files = (str(tmp_path / "a.txt"), str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        _probes_in_a_pool(files)  # os.stat of the missing file, plain
    files = (str(tmp_path / "a.txt"),)
    assert disk_cash.cache(_probes_in_a_pool)(files) == _probes_in_a_pool(files) == ([True], [True], [3], [True])


def _sizes_on_threads(files):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        return list(pool.map(os.path.getsize, files))


def test_a_probe_run_as_a_thread_pool_task_is_an_input_too(disk_cash, tmp_path, recwarn):
    (tmp_path / "a.txt").write_text("abc", encoding="utf-8")
    files = (str(tmp_path / "a.txt"),)
    cached = disk_cash.cache(_sizes_on_threads)
    assert cached(files) == [3]
    (tmp_path / "a.txt").write_text("longer than before", encoding="utf-8")
    assert cached(files) == [18]
