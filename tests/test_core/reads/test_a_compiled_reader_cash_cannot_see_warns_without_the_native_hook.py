"""A cached function that reads a file through a popular library's compiled
reader, with the native file hook off, warns and runs every time.

On Linux and Windows the native file hook sees the file such a reader opens.
On macOS it is not available, and a reader cash does not follow to the file
(``cv2.imread``, ``netCDF4.Dataset``, DuckDB, GDAL through pyogrio, fiona or
osgeo) recorded nothing: after the file changed the old result was served,
with no warning. The call now warns KEY-NATIVE-READ-UNSEEN, naming the
reader, the file and the line, and is not stored, unless the file is named
with ``file_depends_on=``: an entry holding a file nothing can check would be
written on every call and never served.

The hook is switched off here, which is the state macOS is always in; on
macOS the same tests run on the real platform.
"""

from __future__ import annotations

import os
import warnings

import pytest

from cash.tracking import native_reads

gpd = pytest.importorskip("geopandas")
pyogrio = pytest.importorskip("pyogrio")
shapely = pytest.importorskip("shapely")


@pytest.fixture
def hook_off():
    """The native file hook off, as on macOS."""
    was_on = native_reads.active()
    native_reads.disable()
    yield
    if was_on:
        native_reads.enable()


def _write(path, n: int):
    frame = gpd.GeoDataFrame({"n": list(range(n))}, geometry=[shapely.Point(i, i) for i in range(n)])
    frame.to_file(path, driver="GeoJSON")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


def _message(caught, code):
    return next(str(w.message) for w in caught if getattr(w.message, "code", None) == code)


def test_the_call_warns_naming_the_reader_the_file_and_the_line(disk_cash, tmp_path, hook_off):
    path = tmp_path / "points.geojson"
    _write(path, 2)

    @disk_cash.cache
    def count(p):
        return len(pyogrio.read_dataframe(p))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert count(str(path)) == 2
    assert "KEY-NATIVE-READ-UNSEEN" in _codes(caught)
    message = _message(caught, "KEY-NATIVE-READ-UNSEEN")
    assert "pyogrio.read_dataframe" in message
    assert str(path) in message
    assert "return len(pyogrio.read_dataframe(p))" in message


def test_an_edit_to_the_file_is_never_served_stale(disk_cash, tmp_path, hook_off):
    path = tmp_path / "points.geojson"
    _write(path, 2)
    runs: list[int] = []

    @disk_cash.cache
    def count(p):
        runs.append(1)
        return len(gpd.read_file(p))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert count(str(path)) == 2
        assert count(str(path)) == 2
        assert len(runs) == 2  # runs every time
        _write(path, 5)
        assert count(str(path)) == 5


def test_the_call_is_not_stored_and_the_next_miss_says_why(disk_cash, tmp_path, hook_off):
    path = tmp_path / "points.geojson"
    _write(path, 2)

    @disk_cash.cache
    def count(p):
        return len(pyogrio.read_dataframe(p))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            assert count(str(path)) == 2
        why = str(count.explain(str(path)))
    assert count.cache_info()["miss_reasons"] == {"no entry yet": 1, "not stored last time": 2}
    assert "no_entry" in why
    assert f"not stored last time: pyogrio.read_dataframe read {path} in compiled code, which cash cannot check" in why


def test_a_file_named_with_file_depends_on_is_cached_and_checked(disk_cash, tmp_path, hook_off):
    path = tmp_path / "points.geojson"
    _write(path, 2)
    runs: list[int] = []

    @disk_cash.cache(file_depends_on=str(path))
    def count(p):
        runs.append(1)
        return len(pyogrio.read_dataframe(p))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert count(str(path)) == 2
        assert count(str(path)) == 2
    assert "KEY-NATIVE-READ-UNSEEN" not in _codes(caught)
    assert len(runs) == 1
    _write(path, 5)
    assert count(str(path)) == 5


@pytest.mark.skipif(not native_reads.supported()[0], reason="no native file hook on this platform")
def test_with_the_hook_on_the_read_is_seen_and_cached(disk_cash, tmp_path):
    if not native_reads.active():
        pytest.skip("the native file hook is switched off in this run")
    path = tmp_path / "points.geojson"
    _write(path, 2)
    runs: list[int] = []

    @disk_cash.cache
    def count(p):
        runs.append(1)
        return len(pyogrio.read_dataframe(p))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert count(str(path)) == 2
        assert count(str(path)) == 2
    assert "KEY-NATIVE-READ-UNSEEN" not in _codes(caught)
    assert len(runs) == 1
    _write(path, 5)
    assert count(str(path)) == 5
