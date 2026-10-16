"""HDF5, netCDF and raster readers are tracked without the native file hook.

libhdf5, libnetcdf and GDAL open their files in C, so no audit event reports
the read. On Linux and Windows the native file hook (on by default) sees those
opens; on macOS there is no hook, and a cached function over ``h5py.File``
returned the old file's answer after the file changed, with no warning. The
entry points are now wrapped like pyarrow's: ``h5py.File``,
``tables.open_file``, ``rasterio.open`` and xarray's ``open_*`` record the
path they are given. A file opened only to write is an output, not an input.
"""

from __future__ import annotations

import os

import pytest

from cash.tracking import native_reads

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def no_hook():
    """As on macOS: the hook is off, only the wrapped entry points see the read."""
    was = native_reads.active()
    native_reads.disable()
    yield
    if was:
        native_reads.enable()


def _h5(path, value):
    with h5py.File(path, "w") as f:
        f["x"] = np.full(10, value)
    st = os.stat(path)  # an older mtime, so the rewrite below is a change by any rule
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


@pytest.mark.parametrize("how", ["positional", "keyword", "explicit mode"])
def test_an_edited_hdf5_file_recomputes(disk_cash, tmp_path, how):
    path = tmp_path / "a.h5"
    _h5(path, 1)
    runs = []

    def open_it(p):
        if how == "keyword":
            return h5py.File(name=p)
        if how == "explicit mode":
            return h5py.File(p, mode="r")
        return h5py.File(p)

    @disk_cash.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        with open_it(p) as f:
            return int(f["x"][()].sum())

    assert total(str(path)) == 10
    assert total(str(path)) == 10
    assert len(runs) == 1, "an unchanged file recomputed"

    _h5(path, 2)
    assert total(str(path)) == 20, "h5py.File served the old file's answer"
    assert len(runs) == 2


def test_a_file_written_by_the_call_is_not_its_input(disk_cash, tmp_path):
    path = tmp_path / "out.h5"
    runs = []

    @disk_cash.cache(assume_safe=True)
    def write(p, v):
        runs.append(1)
        with h5py.File(p, "w") as f:
            f["x"] = np.full(3, v)
        return v

    write(str(path), 1)
    write(str(path), 1)
    assert len(runs) == 1, "the call's own output made it recompute"


def test_the_read_mode_rule():
    from cash.tracking.reader_patches import _opens_to_read

    assert all(_opens_to_read(m) for m in ("r", "r+", "a", "rb", None))
    assert not any(_opens_to_read(m) for m in ("w", "w-", "x"))
