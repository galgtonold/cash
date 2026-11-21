"""A geometry column is read back with the cyclic collector held off while
shapely makes its geometries, and left as it was found.

Each geometry is an object the collector tracks: making 300,000 of them
from their WKB, the collector walked every object the process holds every
few thousand, the new geometries among them, 30 of the 75 ms the read took.
Nothing it could collect is made meanwhile.
"""

from __future__ import annotations

import gc

import pytest

from cash.backends.serialization import PickleSerializer

gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
shapely = pytest.importorskip("shapely")


def _frame(n=2000):
    """Polygons, which are read back from WKB (plain points are made from
    their coordinates)."""
    rng = np.random.default_rng(0)
    centres = shapely.points(rng.uniform(0, 1, (n, 2)))
    return gpd.GeoDataFrame({"v": np.arange(n)}, geometry=shapely.buffer(centres, 0.01, quad_segs=2), crs="EPSG:4326")


def _stored(frame):
    serializer = PickleSerializer()
    stream, buffers = serializer.serialize_split(frame)
    return lambda: serializer.deserialize_split(stream, buffers)


@pytest.fixture
def collector_seen(monkeypatch):
    """Whether the collector was on each time shapely read WKB."""
    seen = []
    real = shapely.from_wkb

    def from_wkb(*args, **kwargs):
        seen.append(gc.isenabled())
        return real(*args, **kwargs)

    monkeypatch.setattr(shapely, "from_wkb", from_wkb)
    return seen


@pytest.fixture
def collector_restored():
    was = gc.isenabled()
    yield
    if was:
        gc.enable()
    else:
        gc.disable()


def test_the_geometries_are_made_with_the_collector_off_and_it_is_back_on(collector_seen, collector_restored):
    frame = _frame()
    read = _stored(frame)
    gc.enable()
    back = read()
    assert collector_seen == [False]
    assert gc.isenabled()
    assert back.geometry.geom_equals_exact(frame.geometry, 0).all()
    assert back.crs == frame.crs


def test_a_collector_the_program_switched_off_stays_off(collector_seen, collector_restored):
    read = _stored(_frame())
    gc.disable()
    read()
    assert collector_seen == [False]
    assert not gc.isenabled()


def test_the_collector_is_back_on_when_reading_fails(monkeypatch, collector_restored):
    read = _stored(_frame())

    def broken(*args, **kwargs):
        raise ValueError("unreadable WKB")

    monkeypatch.setattr(shapely, "from_wkb", broken)
    gc.enable()
    with pytest.raises(ValueError, match="unreadable WKB"):
        read()
    assert gc.isenabled()
