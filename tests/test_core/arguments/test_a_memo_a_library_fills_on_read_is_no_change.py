"""What pandas and xarray fill in an object as it is read is no change of it.

pandas keeps what a ``cache_readonly`` property computed in the object's
``_cache`` dict: ``df[df.a > 1]`` on a frame subclass (a ``GeoDataFrame``,
a user's own) fills the ``dtype`` and ``fill_value`` of its blocks, and
``da.resample(...)`` fills the ``unit`` of the ``DatetimeArray`` under an
xarray time coordinate. xarray makes a variable's ``attrs`` dict on its first
read. Such an argument is keyed by its pickle, which moved: the call was
taken to change its argument in place and ran every time. The memo is left
out of the key when everything in it is what the class's ``cache_readonly``
properties put there, and xarray's empty dict keys as the None it stood
for. An edit to the object still keys apart, and a call that really changes
it in place runs every time, as plain Python does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.value_hash import compute_hash


class Frame(pd.DataFrame):
    """A frame subclass of the user's: keyed by its pickle, not by cash's
    pandas hasher."""

    _metadata = ["tag"]

    @property
    def _constructor(self):
        return Frame


def _frame() -> Frame:
    return Frame({"a": np.arange(6.0), "b": list("abcdef")})


def _hits(fn) -> int:
    return fn.cache_info()["hits"]


def test_a_frame_subclass_read_by_a_filter_is_stored_and_hit(disk_cash):
    @disk_cash.cache
    def big(frame, x):
        return frame[frame.a > x]

    frame = _frame()
    expected = frame[frame.a > 2]
    pd.testing.assert_frame_equal(big(frame, 2), expected)
    pd.testing.assert_frame_equal(big(frame, 2), expected)
    assert _hits(big) == 1, big.cache_info()["miss_reasons"]
    changed = _frame()
    changed.loc[5, "a"] = -1.0
    pd.testing.assert_frame_equal(big(changed, 2), changed[changed.a > 2])
    assert _hits(big) == 1, "an edited frame must miss"


def test_a_frame_subclass_changed_in_place_runs_every_time(disk_cash):
    @disk_cash.cache
    def bump(frame):
        frame.loc[frame.a > 2, "a"] += 1
        return float(frame.a.sum())

    plain, cached = _frame(), _frame()
    for _ in range(2):
        plain.loc[plain.a > 2, "a"] += 1
        assert bump(cached) == float(plain.a.sum())
    pd.testing.assert_frame_equal(cached, plain)
    assert _hits(bump) == 0


def test_a_value_put_in_the_memo_by_other_code_is_keyed():
    frame = _frame()
    before = compute_hash(frame)
    frame[frame.a > 1]
    assert compute_hash(frame) == before
    frame._mgr.blocks[0]._cache["not_a_property"] = 1
    assert compute_hash(frame) != before


def _array(xr, offset: float = 0.0):
    time = pd.date_range("2000-01-01", periods=120, freq="D")
    return xr.DataArray(np.arange(120.0) + offset, coords={"time": time}, dims="time", name="tas")


@pytest.mark.parametrize("read", ["resample", "dt"])
def test_an_xarray_array_read_through_an_accessor_is_stored_and_hit(disk_cash, read):
    xr = pytest.importorskip("xarray")

    @disk_cash.cache
    def monthly(da):
        if read == "resample":
            return da.resample(time="MS").mean()
        return da.time.dt.month * da

    da = _array(xr)
    expected = monthly.__wrapped__(_array(xr))
    for _ in range(2):
        xr.testing.assert_identical(monthly(da), expected)
    assert _hits(monthly) == 1, monthly.cache_info()["miss_reasons"]
    xr.testing.assert_identical(monthly(_array(xr, 1.0)), monthly.__wrapped__(_array(xr, 1.0)))
    assert _hits(monthly) == 1, "another array must miss"


@pytest.mark.parametrize("change", ["values", "attrs"])
def test_an_xarray_array_changed_in_place_runs_every_time(disk_cash, change):
    xr = pytest.importorskip("xarray")

    def body(da):
        if change == "values":
            da[0] = da[0] + 1
        else:
            da.attrs["n"] = da.attrs.get("n", 0) + 1
        return float(da.resample(time="MS").mean().sum()) + da.attrs.get("n", 0)

    bump = disk_cash.cache(body)
    plain, cached = _array(xr), _array(xr)
    for _ in range(2):
        assert bump(cached) == body(plain)
    xr.testing.assert_identical(cached, plain)
    assert _hits(bump) == 0


def test_a_geodataframe_filtered_is_stored_and_hit(disk_cash):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    def frame(grow: float = 0.0):
        shapes = [Point(i, i).buffer(i + 0.1 + grow) for i in range(5)]
        return gpd.GeoDataFrame({"v": range(5)}, geometry=shapes)

    @disk_cash.cache
    def big(gdf, x):
        return gdf[gdf.area > x]

    gdf = frame()
    expected = frame()[frame().area > 1.0]
    for _ in range(2):
        pd.testing.assert_frame_equal(big(gdf, 1.0), expected)
    assert _hits(big) == 1, big.cache_info()["miss_reasons"]
    other = frame(1.0)
    pd.testing.assert_frame_equal(big(other, 1.0), other[other.area > 1.0])
    assert _hits(big) == 1, "another frame must miss"
