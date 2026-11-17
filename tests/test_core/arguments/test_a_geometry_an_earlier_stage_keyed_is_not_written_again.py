"""A geometry a pipeline hands from stage to stage is turned to WKB once.

A spatial join holds each geometry on every row it matched, and the stages
after it -- filtered, with a column added -- hold the same geometry objects
in new arrays. Each stage's key wrote every row to WKB anew: 0.5 to 1.1 s
per stage for 740,000 joined rows over 195,000 polygons. A shapely geometry
cannot change, so what it adds to a key is remembered by the object, and
the key is still its content: equal geometries in other objects key alike.
"""

from __future__ import annotations

import gc

import pytest

from cash import Cash, FileBackend, content_hashers
from cash.content_hashers import GEOMETRY_PIECES, builtin_hash

gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
shapely = pytest.importorskip("shapely")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _digests():
    """The process's memo of each geometry's piece of a key."""
    return content_hashers.GEOMETRY_DIGESTS


def _polygons(n, seed, radius):
    rng = np.random.default_rng(seed)
    centres = shapely.points(rng.uniform(0, 100, n), rng.uniform(0, 100, n))
    return gpd.GeoDataFrame({"v": np.arange(n)}, geometry=shapely.buffer(centres, radius, quad_segs=4), crs="EPSG:3857")


def _joined():
    left, right = _polygons(800, 0, 1.0), _polygons(400, 1, 3.0).rename(columns={"v": "w"})
    joined = gpd.sjoin(left, right, predicate="intersects")
    # Many-to-many: rows share geometry objects, and stay under the size
    # whose whole array is remembered (`GEOMETRY_PIECES`).
    assert 256 < len(joined) < GEOMETRY_PIECES.MIN_ROWS
    assert len(set(map(id, joined.geometry.values._data))) < len(joined)
    return left, joined


@pytest.fixture
def written_rows(monkeypatch):
    """How many geometries `shapely.to_wkb` writes, call by call."""
    memo = getattr(content_hashers, "GEOMETRY_DIGESTS", None)
    if memo is not None:
        memo.clear()
    GEOMETRY_PIECES.clear()
    counts = []
    to_wkb = shapely.to_wkb

    def counting(geometry, *args, **kwargs):
        counts.append(np.size(geometry))
        return to_wkb(geometry, *args, **kwargs)

    monkeypatch.setattr(shapely, "to_wkb", counting)
    yield counts
    if memo is not None:
        memo.clear()


def test_the_stages_after_a_join_write_no_geometry_again(written_rows):
    left, joined = _joined()
    builtin_hash(left)
    assert sum(written_rows) == len(left)
    written_rows.clear()
    filtered = joined[joined["w"] % 2 == 0]
    added = filtered.assign(twice=filtered["v"] * 2)
    for stage in (joined, filtered, added, added.reset_index(drop=True)):
        builtin_hash(stage)
    assert sum(written_rows) == 0


def test_a_join_writes_each_geometry_it_holds_once(written_rows):
    _, joined = _joined()
    builtin_hash(joined)
    assert 0 < sum(written_rows) == len(set(map(id, joined.geometry.values._data))) < len(joined)


def test_equal_geometries_in_other_objects_key_alike(written_rows):
    left, joined = _joined()
    builtin_hash(left)
    fresh = joined.copy()
    fresh.geometry.values._data[:] = shapely.from_wkb(shapely.to_wkb(joined.geometry.values._data))
    assert not set(map(id, fresh.geometry.values._data)) & set(map(id, joined.geometry.values._data))
    assert builtin_hash(fresh) == builtin_hash(joined)
    _digests().clear()
    assert builtin_hash(joined) == builtin_hash(fresh)


@pytest.mark.parametrize("change", ["one coordinate", "srid", "a missing row", "two rows swapped"])
def test_a_changed_geometry_keys_apart(written_rows, change):
    _, joined = _joined()
    before = builtin_hash(joined)
    data = joined.geometry.values._data
    changed = joined.copy()
    new = changed.geometry.values._data
    if change == "one coordinate":
        new[7] = shapely.affinity.translate(data[7], 1e-9)
    elif change == "srid":
        new[7] = shapely.set_srid(data[7], 4326)
    elif change == "a missing row":
        new[7] = None
    else:
        first, second = next((i, j) for i in range(len(data)) for j in range(i + 1, len(data)) if data[i] != data[j])
        new[first], new[second] = data[second], data[first]
    assert builtin_hash(changed) != before
    assert builtin_hash(joined) == before


def test_points_key_as_before_without_the_memo(written_rows):
    rng = np.random.default_rng(3)
    points = gpd.GeoSeries(gpd.points_from_xy(rng.uniform(size=1000), rng.uniform(size=1000)), crs="EPSG:4326")
    again = gpd.GeoSeries(shapely.from_wkb(shapely.to_wkb(points.values._data)), crs="EPSG:4326")
    assert builtin_hash(points) == builtin_hash(again)
    assert len(_digests()) == 0  # a point is its own WKB: nothing to remember


def test_a_full_memo_starts_again(written_rows, monkeypatch):
    left, joined = _joined()
    monkeypatch.setattr(_digests(), "MAX_HELD", len(left) + 10)
    before = builtin_hash(left)
    assert len(_digests()) == len(left)
    other = _polygons(400, 5, 1.0)
    builtin_hash(other)
    assert len(_digests()) == len(other)
    assert builtin_hash(left) == before


def test_a_cached_pipeline_hits_on_equal_geometries(tmp_path, written_rows):
    cache = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @cache.cache
    def area(frame):
        return float(frame.geometry.area.sum())

    left, joined = _joined()
    assert area(joined) == float(joined.geometry.area.sum())
    copy = joined.copy()
    copy.geometry.values._data[:] = shapely.from_wkb(shapely.to_wkb(joined.geometry.values._data))
    assert area(copy) == area(joined)
    assert area.cache_info()["hits"] == 2
    moved = joined.copy()
    moved.geometry.values._data[0] = shapely.affinity.translate(joined.geometry.values._data[0], 1.0)
    area(moved)
    assert area.cache_info()["misses"] == 2


def test_a_deleted_frame_takes_its_geometries_with_it(written_rows):
    left, joined = _joined()
    builtin_hash(joined)
    before = len(_digests())
    del left, joined
    gc.collect()
    other = _polygons(300, 7, 1.0)
    builtin_hash(other)
    # Only the new frame's geometries are remembered: the deleted ones
    # were held weakly, and their memory went with them.
    assert len(_digests()) == len(other) < before


def test_geometries_made_where_deleted_ones_were_key_by_their_own_content(written_rows):
    for seed in range(4):
        frame = _polygons(400, seed, 1.0 + seed)
        key = builtin_hash(frame)
        copy = frame.copy()
        copy.geometry.values._data[:] = shapely.from_wkb(shapely.to_wkb(frame.geometry.values._data))
        assert builtin_hash(copy) == key
        del frame, copy
        gc.collect()
    # Each round's geometries may sit where the last round's were: keyed
    # with a fresh memo, they key alike.
    keys = [builtin_hash(_polygons(400, seed, 1.0 + seed)) for seed in range(4)]
    _digests().clear()
    assert [builtin_hash(_polygons(400, seed, 1.0 + seed)) for seed in range(4)] == keys
