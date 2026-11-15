"""A geopandas frame or series keys by its columns, its geometries (as
WKB, every dimension), its CRS and its active geometry column.

Pickled whole, a frame restored from the cache pickled its column labels'
state in another order than the frame it was stored from, so the next
cached function missed on every run; and a frame of 200,000 points was read
a geometry at a time by the check that a result shares nothing with a
function's closure, past its time, and never stored.
"""

from __future__ import annotations

import pickle
import warnings

import pytest

from cash import Cash, FileBackend
from cash.content_hashers import GEOMETRY_PIECES, builtin_hash

gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
shapely = pytest.importorskip("shapely")


def _points(n=50, seed=0, crs="EPSG:3857"):
    rng = np.random.default_rng(seed)
    return gpd.GeoDataFrame(
        {"val": rng.normal(size=n)},
        geometry=gpd.points_from_xy(rng.uniform(0, 100, n), rng.uniform(0, 100, n)),
        crs=crs,
    )


def _cache(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _summary(frame):
    return (len(frame), float(frame.geometry.x.sum()), str(frame.crs), float(frame["val"].sum()))


def test_a_restored_frame_keys_like_the_frame_it_was_stored_from(tmp_path):
    frame = _points()
    restored = pickle.loads(pickle.dumps(frame))
    assert builtin_hash(restored) == builtin_hash(frame)
    f = _cache(tmp_path).cache(_summary)
    assert f(frame) == _summary(frame)
    assert f(restored) == _summary(frame)
    assert f.cache_info()["hits"] == 1
    Cash(backend=FileBackend(cache_dir=str(tmp_path / "made"))).cache(_points)()
    from_disk = Cash(backend=FileBackend(cache_dir=str(tmp_path / "made"))).cache(_points)()
    assert f(from_disk) == _summary(frame)
    assert f.cache_info()["hits"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "coordinate",
        "z",
        "crs",
        "crs_in_place",
        "column",
        "geometry_type",
        "missing",
    ],
)
def test_a_changed_frame_misses(tmp_path, change):
    frame = _points()
    other = frame.copy()
    if change == "coordinate":
        other.loc[3, "geometry"] = shapely.Point(other.geometry.x[3] + 1e-9, other.geometry.y[3])
    elif change == "z":
        frame.loc[3, "geometry"] = shapely.Point(1, 2, 3)
        other.loc[3, "geometry"] = shapely.Point(1, 2, 4)
    elif change == "crs":
        other = other.set_crs("EPSG:4326", allow_override=True)
    elif change == "crs_in_place":
        other.set_crs("EPSG:4326", allow_override=True, inplace=True)
    elif change == "column":
        other.loc[3, "val"] += 1.0
    elif change == "geometry_type":
        frame.loc[3, "geometry"] = shapely.Point(1, 2)
        other.loc[3, "geometry"] = shapely.MultiPoint([(1, 2)])
    elif change == "missing":
        other.loc[3, "geometry"] = None
    assert builtin_hash(frame) != builtin_hash(other)
    calls = []

    def plain(gdf):
        return gdf.geometry.to_wkb(include_srid=True).tolist(), str(gdf.crs), gdf["val"].tolist()

    def body(gdf):
        calls.append(1)
        return plain(gdf)

    f = _cache(tmp_path).cache(body)
    assert f(frame) == plain(frame)
    assert f(other) == plain(other)
    assert calls == [1, 1]


def test_the_active_geometry_column_reaches_the_key():
    frame = _points()
    frame["other"] = frame.geometry.buffer(1.0)
    assert builtin_hash(frame) != builtin_hash(frame.set_geometry("other"))


def test_a_geoseries_keys_by_its_geometries_and_crs():
    series = _points().geometry
    assert builtin_hash(series) == builtin_hash(pickle.loads(pickle.dumps(series)))
    assert builtin_hash(series) != builtin_hash(series.set_crs("EPSG:4326", allow_override=True))
    assert builtin_hash(series) != builtin_hash(series.translate(1e-6))


def test_a_big_geometry_array_written_in_place_is_read_again():
    frame = _points(n=GEOMETRY_PIECES.MIN_ROWS + 10)
    first = builtin_hash(frame)
    assert builtin_hash(frame) == first  # from the memo
    frame.geometry.values._data[7] = shapely.Point(-1, -1)  # under pandas, into the array
    assert builtin_hash(frame) != first
    frame.loc[7, "geometry"] = frame.loc[8, "geometry"]
    second = builtin_hash(frame)
    fresh = pickle.loads(pickle.dumps(frame))
    GEOMETRY_PIECES.clear()
    assert builtin_hash(frame) == second == builtin_hash(fresh)


def test_a_cached_call_on_a_big_frame_hits_and_misses_like_plain_python(tmp_path):
    frame = _points(n=GEOMETRY_PIECES.MIN_ROWS + 10)
    calls = []

    def body(gdf):
        calls.append(1)
        return _summary(gdf)

    f = _cache(tmp_path).cache(body)
    assert f(frame) == _summary(frame)
    assert f(frame) == _summary(frame)
    assert calls == [1]
    frame.loc[0, "geometry"] = shapely.Point(500, 500)
    assert f(frame) == _summary(frame)
    assert calls == [1, 1]


def _many_points(n):
    return _points(n=n)


def test_a_frame_of_many_points_is_stored(tmp_path):
    f = _cache(tmp_path).cache(_many_points)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = f(200_000)
    assert f(200_000).geometry.equals(first.geometry)
    assert f.cache_info()["hits"] == 1
    assert [w["code"] for w in f.cache_info()["warnings"]] == []
