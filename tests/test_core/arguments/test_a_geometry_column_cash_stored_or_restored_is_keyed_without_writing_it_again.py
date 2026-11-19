"""A geometry column cash has just stored or restored keys from the WKB in hand.

Storing a geopandas frame writes each geometry to WKB, and restoring one
reads it back from WKB. The next cached function the frame was handed to
then wrote every geometry to WKB once more for its key: 0.08 s of 0.13 s
per 300,000 points, on every stage of a pipeline restored from the cache.
The key is still the one a fresh read gives, a frame changed since keys
apart, and a geometry with an SRID (which that WKB leaves out) is read
again.
"""

from __future__ import annotations

import textwrap

import pytest

from cash.backends.serialization import PickleSerializer
from cash.content_hashers import GEOMETRY_DIGESTS, GEOMETRY_PIECES, builtin_hash
from tests._scripts import run_python

gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
shapely = pytest.importorskip("shapely")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

ROWS = 20_000


def _points(missing=False):
    rng = np.random.default_rng(0)
    geometry = list(shapely.points(rng.uniform(0, 10, ROWS), rng.uniform(0, 10, ROWS)))
    if missing:
        geometry[::7] = [None] * len(geometry[::7])
    return gpd.GeoDataFrame({"v": np.arange(ROWS)}, geometry=geometry, crs="EPSG:4326")


def _mixed():
    """Points, polygons (WKB too long to be its own piece), an empty and a
    3-d point, missing rows, and rows sharing one object."""
    frame = _points(missing=True)
    geometry = frame.geometry.values._data.copy()
    geometry[3::11] = shapely.buffer(geometry[3::11], 0.5, quad_segs=2)
    geometry[5] = shapely.Point()
    geometry[6] = shapely.Point(1, 2, 3)
    geometry[ROWS // 2 :] = geometry[: ROWS - ROWS // 2]
    return gpd.GeoDataFrame({"v": np.arange(ROWS)}, geometry=geometry, crs="EPSG:3857")


FRAMES = {"points": _points, "points with missing rows": lambda: _points(missing=True), "mixed": _mixed}


@pytest.fixture
def written_rows(monkeypatch):
    """How many geometries `shapely.to_wkb` writes, with nothing remembered."""
    GEOMETRY_DIGESTS.clear()
    GEOMETRY_PIECES.clear()
    counts = []
    to_wkb = shapely.to_wkb

    def counting(geometry, *args, **kwargs):
        counts.append(np.size(geometry))
        return to_wkb(geometry, *args, **kwargs)

    monkeypatch.setattr(shapely, "to_wkb", counting)
    yield counts
    GEOMETRY_DIGESTS.clear()
    GEOMETRY_PIECES.clear()


def _fresh_key(frame):
    GEOMETRY_DIGESTS.clear()
    GEOMETRY_PIECES.clear()
    return builtin_hash(frame)


def _round_trip(frame):
    serializer = PickleSerializer()
    return serializer.deserialize_split(*serializer.serialize_split(frame))


@pytest.mark.parametrize("make", FRAMES.values(), ids=FRAMES.keys())
def test_a_restored_frame_keys_as_a_fresh_read_without_writing_wkb(written_rows, make):
    frame = make()
    restored = _round_trip(frame)
    written_rows.clear()
    key = builtin_hash(restored)
    assert sum(written_rows) == 0
    assert key == _fresh_key(restored) == _fresh_key(frame)


@pytest.mark.parametrize("make", FRAMES.values(), ids=FRAMES.keys())
def test_a_stored_frame_keys_as_a_fresh_read_without_writing_wkb_again(written_rows, make):
    frame = make()
    PickleSerializer().serialize_split(frame)
    stored = sum(written_rows)
    key = builtin_hash(frame)
    assert sum(written_rows) == stored
    assert key == _fresh_key(frame)


def test_a_geometry_with_an_srid_is_read_again_and_keeps_it_in_the_key(written_rows):
    frame = _points()
    frame.geometry.values._data[:] = shapely.set_srid(frame.geometry.values._data, 4326)
    PickleSerializer().serialize_split(frame)
    stored = sum(written_rows)
    key = builtin_hash(frame)
    assert sum(written_rows) > stored
    assert key == _fresh_key(frame) != _fresh_key(_points())


def test_a_restored_frame_changed_since_keys_apart(written_rows):
    restored = _round_trip(_points())
    before = builtin_hash(restored)
    restored.geometry.values._data[7] = shapely.Point(-1, -1)
    assert builtin_hash(restored) != before
    assert builtin_hash(restored) == _fresh_key(restored)


def test_a_pipeline_restored_in_a_new_process_keys_its_next_stage_without_writing_wkb(tmp_path):
    script = tmp_path / "pipeline.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import numpy as np, geopandas as gpd, shapely, cash

            written = []
            to_wkb = shapely.to_wkb

            def counting(geometry, *args, **kwargs):
                written.append(np.size(geometry))
                return to_wkb(geometry, *args, **kwargs)

            shapely.to_wkb = counting

            @cash.cache
            def load(n):
                rng = np.random.default_rng(0)
                return gpd.GeoDataFrame(
                    {{"v": np.arange(n)}}, geometry=gpd.points_from_xy(rng.uniform(size=n), rng.uniform(size=n)), crs=4326
                )

            @cash.cache
            def total(frame):
                return float(frame.geometry.x.sum())

            print(total(load({ROWS})), load.cache_info()["hits"], total.cache_info()["hits"], sum(written))
            """
        ),
        encoding="utf-8",
    )
    first = run_python(script, cwd=tmp_path).stdout.split()
    second = run_python(script, cwd=tmp_path).stdout.split()
    # The first process writes each geometry once, to store it.
    assert first == [first[0], "0", "0", str(ROWS)]
    assert second == [first[0], "1", "1", "0"]
