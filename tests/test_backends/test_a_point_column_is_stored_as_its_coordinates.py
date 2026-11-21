"""A column of plain points is stored as its coordinates, not as WKB.

A point with two coordinates, no Z or M, not empty and without an SRID is
its x and y and nothing else. Writing 300,000 of them to WKB to store a
frame took 0.15 s, and reading them back from WKB 0.1 s; they are now
read and made again from their coordinates in C. They read back exactly
as geopandas' own pickle reads them, with missing rows and rows sharing
one object as they were, and they key as the WKB they stand for. Any
other column -- a point with Z, M, an SRID, a NaN coordinate or none at
all, or any other geometry among them -- is stored as WKB, as before.
"""

from __future__ import annotations

import gc
import pickle

import pytest

from cash import Cash, FileBackend
from cash.backends.serialization import PickleSerializer
from cash.content_hashers import GEOMETRY_DIGESTS, GEOMETRY_PIECES, builtin_hash

gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
shapely = pytest.importorskip("shapely")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

ROWS = 5000

#: Coordinates whose bits a round trip must keep.
AWKWARD = [[-0.0, 0.0], [np.inf, -np.inf], [5e-324, -1e308], [0.1, 1 / 3]]


def _points(n=ROWS, crs="EPSG:4326"):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-180, 180, (n, 2))
    xy[: len(AWKWARD)] = AWKWARD
    return gpd.GeoDataFrame({"v": np.arange(n)}, geometry=shapely.points(xy), crs=crs)


def _round_trip(value):
    serializer = PickleSerializer()
    return serializer.deserialize_split(*serializer.serialize_split(value))


@pytest.fixture
def wkb_calls(monkeypatch):
    """How many geometries shapely writes to or reads from WKB."""
    _round_trip(_points())  # anything done once per process, done
    calls = {"to": 0, "from": 0}
    to_wkb, from_wkb = shapely.to_wkb, shapely.from_wkb

    def counting_to(geometry, *args, **kwargs):
        calls["to"] += int(np.size(geometry))
        return to_wkb(geometry, *args, **kwargs)

    def counting_from(geometry, *args, **kwargs):
        calls["from"] += int(np.size(geometry))
        return from_wkb(geometry, *args, **kwargs)

    monkeypatch.setattr(shapely, "to_wkb", counting_to)
    monkeypatch.setattr(shapely, "from_wkb", counting_from)
    return calls


def _coordinates(series):
    present = series.values._data[~series.isna().to_numpy()]
    return np.column_stack([shapely.get_x(present), shapely.get_y(present)])


def _assert_as_plain_pickle_reads(back, frame):
    plain = pickle.loads(pickle.dumps(frame))
    assert type(back) is type(frame)
    assert back.crs == plain.crs
    assert back.geometry.dtype == plain.geometry.dtype
    assert back.geometry.isna().tolist() == plain.geometry.isna().tolist()
    assert (shapely.to_wkb(back.geometry.values._data) == shapely.to_wkb(plain.geometry.values._data)).all()
    assert back.drop(columns="geometry").equals(frame.drop(columns="geometry"))


def test_points_round_trip_bit_for_bit_without_wkb(wkb_calls):
    frame = _points()
    back = _round_trip(frame)
    assert wkb_calls == {"to": 0, "from": 0}
    _assert_as_plain_pickle_reads(back, frame)
    assert _coordinates(back.geometry).tobytes() == _coordinates(frame.geometry).tobytes()
    assert np.signbit(_coordinates(back.geometry)[0, 0])


def test_missing_rows_and_rows_sharing_a_point_read_back_as_they_were(wkb_calls):
    frame = _points()
    data = frame.geometry.values._data
    data[[5, 6, ROWS - 1]] = None
    data[100:200] = data[300]
    back = _round_trip(frame)
    assert wkb_calls == {"to": 0, "from": 0}
    _assert_as_plain_pickle_reads(back, frame)
    held = back.geometry.values._data
    assert len({id(item) for item in held[100:200]} | {id(held[300])}) == 1


def test_points_are_made_with_the_collector_off(monkeypatch):
    stored = PickleSerializer().serialize_split(_points())
    seen = []
    real = shapely.points

    def points(*args, **kwargs):
        seen.append(gc.isenabled())
        return real(*args, **kwargs)

    monkeypatch.setattr(shapely, "points", points)
    was = gc.isenabled()
    gc.enable()
    try:
        PickleSerializer().deserialize_split(*stored)
        assert seen == [False] and gc.isenabled()
    finally:
        if not was:
            gc.disable()


def _with(frame, row, geometry):
    frame.geometry.values._data[row] = geometry
    return frame


OTHERS = {
    "a 3-d point": lambda f: _with(f, 9, shapely.Point(1, 2, 3)),
    "a measured point": lambda f: _with(f, 9, shapely.from_wkt("POINT M (1 2 3)")),
    "an empty point": lambda f: _with(f, 9, shapely.Point()),
    "a NaN coordinate": lambda f: _with(f, 9, shapely.points([np.nan, 1.0])),
    "a polygon": lambda f: _with(f, 9, shapely.box(0, 0, 1, 1)),
    "an SRID": lambda f: _with(f, slice(None), shapely.set_srid(f.geometry.values._data, 4326)),
}


@pytest.mark.parametrize("change", OTHERS.values(), ids=OTHERS.keys())
def test_any_other_column_is_stored_as_wkb_and_reads_back_as_before(wkb_calls, change):
    frame = change(_points())
    back = _round_trip(frame)
    assert wkb_calls["to"] > 0 and wkb_calls["from"] > 0
    _assert_as_plain_pickle_reads(back, frame)


def test_points_key_as_the_wkb_they_stand_for(wkb_calls):
    frame = _points()
    data = frame.geometry.values._data
    assert GEOMETRY_DIGESTS.joined(data) == b"".join(shapely.to_wkb(data, hex=False, include_srid=True))
    wkb_calls.update(to=0)
    GEOMETRY_DIGESTS.clear()
    GEOMETRY_PIECES.clear()
    key = builtin_hash(frame)
    assert wkb_calls["to"] == 0
    moved = _points()
    moved.geometry.values._data[7] = shapely.Point(-1, -1)
    assert builtin_hash(moved) != key
    GEOMETRY_PIECES.clear()
    assert builtin_hash(_round_trip(frame)) == key


def test_a_cached_point_frame_is_read_back_from_disk_equal(tmp_path):
    def load(n):
        return _points(n)

    made = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(load)(ROWS)
    again = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(load)
    back = again(ROWS)
    assert again.cache_info()["hits"] == 1
    _assert_as_plain_pickle_reads(back, made)
    assert _coordinates(back.geometry).tobytes() == _coordinates(made.geometry).tobytes()
