"""A geometry column is stored as each geometry object it holds, once.

A spatial join holds each geometry on every row it matched; geopandas
pickles one WKB per row, so 740,000 joined rows over 195,000 polygons took
1.26 s and 240 MB to store. Each distinct object is now written once with
which rows hold it, and read back equal, with the same CRS and dtype, and
with rows that shared an object sharing one again.
"""

from __future__ import annotations

import pickle

import pytest

from cash import Cash, FileBackend
from cash.backends.serialization import PickleSerializer

gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
shapely = pytest.importorskip("shapely")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _polygons(n, seed, radius, crs="EPSG:3857"):
    rng = np.random.default_rng(seed)
    centres = shapely.points(rng.uniform(0, 100, n), rng.uniform(0, 100, n))
    return gpd.GeoDataFrame({"v": np.arange(n)}, geometry=shapely.buffer(centres, radius, quad_segs=4), crs=crs)


def _joined():
    left, right = _polygons(800, 0, 1.0), _polygons(400, 1, 3.0).rename(columns={"v": "w"})
    joined = gpd.sjoin(left, right, predicate="intersects")
    assert len(set(map(id, joined.geometry.values._data))) < len(joined)
    return joined


def _round_trip(value):
    serializer = PickleSerializer()
    stream, buffers = serializer.serialize_split(value)
    return serializer.deserialize_split(stream, buffers), len(stream) + sum(memoryview(b).nbytes for b in buffers)


def _sharing(series):
    """Which rows hold one object: each row's first row holding its object."""
    first = {}
    return [first.setdefault(id(item), row) for row, item in enumerate(series.values._data)]


def _assert_same(back, frame):
    assert type(back) is type(frame)
    assert back.crs == frame.crs
    assert back.geometry.dtype == frame.geometry.dtype
    assert list(back.columns) == list(frame.columns)
    assert back.geometry.isna().tolist() == frame.geometry.isna().tolist()
    present = ~frame.geometry.isna()
    assert back.geometry[present].geom_equals_exact(frame.geometry[present], 0).all()
    assert (shapely.to_wkb(back.geometry.values._data) == shapely.to_wkb(frame.geometry.values._data)).all()
    plain = pickle.loads(pickle.dumps(frame))
    assert (shapely.to_wkb(back.geometry.values._data) == shapely.to_wkb(plain.geometry.values._data)).all()
    assert back.drop(columns=frame.geometry.name).equals(frame.drop(columns=frame.geometry.name))


def test_a_joined_frame_writes_each_geometry_object_once(monkeypatch):
    joined = _joined()
    distinct = len(set(map(id, joined.geometry.values._data)))
    counts = []
    to_wkb = shapely.to_wkb

    def counting(geometry, *args, **kwargs):
        counts.append(np.size(geometry))
        return to_wkb(geometry, *args, **kwargs)

    monkeypatch.setattr(shapely, "to_wkb", counting)
    _, size = _round_trip(joined)
    assert sum(counts) == distinct
    monkeypatch.setattr(shapely, "to_wkb", to_wkb)
    whole = sum(map(len, to_wkb(joined.geometry.values._data)))
    assert size < whole * distinct / len(joined) + 64 * len(joined)


def test_a_joined_frame_reads_back_equal_with_its_rows_sharing_as_they_did():
    joined = _joined()
    back, _ = _round_trip(joined)
    _assert_same(back, joined)
    assert _sharing(back.geometry) == _sharing(joined.geometry)


@pytest.mark.parametrize(
    "kind",
    [
        "no repeats",
        "missing rows",
        "z and m",
        "empty geometries",
        "no crs",
        "mixed types",
        "points ending in zero bytes",
    ],
)
def test_a_geometry_column_reads_back_as_plain_pickle_reads_it(kind):
    frame = _polygons(300, 2, 1.0, crs=None if kind == "no crs" else "EPSG:4326")
    data = frame.geometry.values._data
    if kind == "missing rows":
        data[[0, 5, 299]] = None
        data[10] = data[11]
    elif kind == "z and m":
        data[:] = shapely.from_wkt(["POINT ZM (1 2 3 4)", "LINESTRING Z (0 0 1, 1 1 2)", "POINT M (5 6 7)"] * 100)
    elif kind == "empty geometries":
        data[[1, 2]] = shapely.from_wkt(["POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY"])
    elif kind == "mixed types":
        data[::3] = shapely.points(np.arange(100), np.arange(100))
        data[1::3] = shapely.linestrings([[[0, 0], [i, 1]] for i in range(100)])
    elif kind == "points ending in zero bytes":
        # One WKB length for every row, read back by numpy: a coordinate of
        # 0.0 ends a point's WKB in eight zero bytes, which must stay.
        data[:] = shapely.points(np.arange(300) % 3, np.zeros(300))
        data[7] = None
    back, _ = _round_trip(frame)
    _assert_same(back, frame)
    assert _sharing(back.geometry) == _sharing(frame.geometry)


def test_a_geometry_series_and_an_empty_frame_read_back_equal():
    series = _joined().geometry
    back, _ = _round_trip(series)
    assert type(back) is type(series) and back.crs == series.crs and back.name == series.name
    assert back.geom_equals_exact(series, 0).all() and back.index.equals(series.index)
    empty = _polygons(0, 0, 1.0)
    back, _ = _round_trip(empty)
    assert len(back) == 0 and back.crs == empty.crs and back.geometry.dtype == empty.geometry.dtype


def test_a_cached_join_is_read_back_from_disk_equal(tmp_path):
    def join(n):
        return _joined()

    first = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(join)
    made = first(1)
    again = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(join)
    back = again(1)
    assert again.cache_info()["hits"] == 1
    _assert_same(back, made)
    assert _sharing(back.geometry) == _sharing(made.geometry)
