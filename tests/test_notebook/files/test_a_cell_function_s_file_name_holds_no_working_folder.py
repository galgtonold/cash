"""A function defined in a cell names no file under the working folder.

Cash compiled each statement under ``<cash-{digest}>``, a relative name. A
library that makes a function's file absolute joined it to the working folder:
joblib's ``Memory`` named its cache folder after it, so the working folder
stood in every path twice, and in a deep folder on Windows every
``output.pkl`` passed 260 characters and failed to write. ``Parallel`` workers
computed every call again, on every round. A statement's name now stands at
the root of the drive Python is installed on (``/<cash-{digest}>``,
``C:\\<cash-{digest}>``), which is never a file and does not move with the
working folder, and a helper's key is what it was.
"""

from __future__ import annotations

import os

import pytest

import cash._paths
import cash.notebook.compiled_source
import cash.tracking.reader_patches
from cash.decorator.function_identity import hash_callable_source
from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

HELPER = '''import time
def costly(data, column):
    """Sleep a little, then return a column."""
    time.sleep(0.01)
    return [row[column] for row in data]
'''


@pytest.fixture
def deep_folder(tmp_path, monkeypatch):
    folder = tmp_path / ("working_folder_" + "d" * 40) / ("e" * 40)
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


def _define(cash_magics, cell=HELPER):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cash_magics.shell.user_ns["__name__"] = "__main__"  # as in a kernel: joblib names __main__'s functions by file
    run_cash_cell(cash_magics, cell, cells=[cell])
    return cash_magics.shell.user_ns["costly"]


def test_a_cell_function_s_file_is_absolute_and_no_file(cash_magics, deep_folder):
    fn = _define(cash_magics)
    name = fn.__code__.co_filename
    assert os.path.isabs(name), name
    assert os.path.abspath(name) == name
    assert "working_folder_" not in name
    assert not os.path.exists(name)


def test_joblib_s_folder_for_a_cell_function_holds_no_working_folder(cash_magics, deep_folder):
    func_inspect = pytest.importorskip("joblib.func_inspect")
    fn = _define(cash_magics)
    modules, name = func_inspect.get_func_name(fn)
    folder = "-".join(modules)
    assert name == "costly"
    assert modules[0].startswith("__main__-"), modules
    assert "working_folder_" not in folder and "e" * 40 not in folder, folder


def test_joblib_workers_store_a_cell_function_s_results_from_a_deep_folder(cash_magics, deep_folder, tmp_path):
    pytest.importorskip("joblib")
    location = tmp_path / "jl"
    cell = (
        HELPER
        + f"""from joblib import Memory, Parallel, delayed
costly_cached = Memory({str(location)!r}, verbose=0).cache(costly)
def mean_of(data, column):
    values = costly_cached(data, column)
    return sum(values) / len(values)
"""
    )
    _define(cash_magics, cell)
    run = "results = Parallel(n_jobs=2)(delayed(mean_of)([[1, 2], [3, 4]], c) for c in range(2))"
    run_cash_cell(cash_magics, run, cells=[cell, run])
    assert cash_magics.shell.user_ns["results"] == [2.0, 3.0]
    outputs = sorted(p.parent.name for p in location.rglob("output.pkl"))
    assert len(outputs) == 2, sorted(str(p.relative_to(location)) for p in location.rglob("*"))
    for path in location.rglob("*"):
        assert "working_folder_" not in str(path.relative_to(location))


def test_a_cell_helper_s_key_is_what_it_was_under_the_relative_name(cash_magics, monkeypatch):
    now = hash_callable_source(_define(cash_magics))
    cash_magics.shell.user_ns.pop("costly")
    for module, attribute in (
        (cash._paths, "CELL_NAME_PREFIX"),
        (cash.notebook.compiled_source, "CELL_NAME_PREFIX"),
        (cash.tracking.reader_patches, "_NOTEBOOK_STATEMENT_PREFIX"),
    ):
        monkeypatch.setattr(module, attribute, "<cash-")
    before = _define(cash_magics, HELPER + "\n")
    assert before.__code__.co_filename.startswith("<cash-")
    assert hash_callable_source(before) == now


def test_a_cell_function_s_source_survives_a_linecache_check(cash_magics, deep_folder):
    """IPython and ``inspect`` call ``linecache.checkcache()``, which drops an
    entry whose name looks like a path and fails ``os.stat``, unless it was
    registered without a modification time."""
    import inspect
    import linecache

    fn = _define(cash_magics)
    linecache.checkcache()
    linecache.checkcache(fn.__code__.co_filename)
    assert "def costly(data, column):" in inspect.getsource(fn)


@pytest.mark.parametrize(
    "base_prefix, root",
    [
        (r"C:\Python314", "C:\\"),
        (r"D:\envs\py314", "D:\\"),
        (r"\\server\share\py314", "\\\\server\\share\\"),
    ],
)
def test_the_name_stands_on_python_s_drive_whatever_the_working_folder(base_prefix, root):
    import ntpath

    assert cash._paths.cell_root(base_prefix, ntpath) == root


def test_the_name_does_not_move_with_the_working_folder(cash_magics, tmp_path, monkeypatch):
    names = []
    for sub in ("a", "b" * 30):
        (tmp_path / sub).mkdir()
        monkeypatch.chdir(tmp_path / sub)
        names.append(_define(cash_magics).__code__.co_filename)
        cash_magics.shell.user_ns.pop("costly")
    assert names[0] == names[1]
    assert names[0].startswith(cash._paths.CELL_ROOT)
