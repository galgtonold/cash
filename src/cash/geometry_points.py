"""A column of plain shapely points as its coordinates, and their WKB.

geopandas' points (``points_from_xy``, a frame read from x/y columns) are
the most common geometry column, and the one cash wrote to WKB to store
it and to key it, and read back from WKB to restore it: about 0.1 s each
way per 300,000 points. A plain point -- two coordinates, no Z or M, not
empty, no SRID -- is its two coordinates and nothing else, so its
coordinates are read in C (``get_x``, ``get_y``), it is made again from them in C
(``shapely.points``), and its WKB is laid out from them by numpy. Any
other geometry, or a point a coordinate of which is NaN (whose WKB is an
empty point's), is left to WKB.

The WKB is shapely's (``to_wkb`` with its defaults, and with
``include_srid`` for a point without one: the same bytes). Each process
checks that once, on points chosen to catch the byte order, a negative
zero, infinities and subnormals (`_layout_holds`); should shapely ever
write another layout, every point goes the WKB way again.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

#: Bytes in the WKB of a 2-d point: byte order, type, x, y.
POINT_WKB_BYTES = 21

#: ``(the shapely module checked, whether the layout held)``.
_CHECKED: tuple[Any, bool] | None = None
_CHECK_LOCK = threading.Lock()


def plain_point_coordinates(geometries: Any) -> Any:
    """The ``(n, 2)`` float64 x and y of the 1-d object array
    *geometries* when every item of it is a plain point (see the module
    doc), else None. None for an empty array, and for one holding
    anything but shapely geometries."""
    import numpy as np
    import shapely

    if type(geometries) is not np.ndarray or geometries.ndim != 1 or not len(geometries):
        return None
    try:
        types = shapely.get_type_id(geometries)
    except (TypeError, ValueError):  # not geometries
        return None
    if (types != 0).any():  # a missing row (-1) or another type
        return None
    if (shapely.get_coordinate_dimension(geometries) != 2).any():  # Z or M
        return None
    if shapely.is_empty(geometries).any() or shapely.get_srid(geometries).any():
        return None
    coordinates = _read_coordinates(geometries)
    if np.isnan(coordinates).any() or not _layout_holds():
        return None
    return coordinates


def _read_coordinates(points: Any) -> Any:
    """The ``(n, 2)`` x and y of the 1-d object array *points*, all 2-d
    points: ``get_x`` and ``get_y`` read them in half the time
    ``get_coordinates`` takes."""
    import numpy as np
    import shapely

    coordinates = np.empty((len(points), 2), dtype=np.float64)
    coordinates[:, 0] = shapely.get_x(points)
    coordinates[:, 1] = shapely.get_y(points)
    return coordinates


def points_from_coordinates(coordinates: Any) -> Any:
    """A 1-d object array of the points `plain_point_coordinates` read."""
    import shapely

    return shapely.points(coordinates)


def point_wkb(coordinates: Any) -> bytes:
    """The WKB of the points at *coordinates* (`plain_point_coordinates`),
    one after another, as shapely writes them."""
    import numpy as np

    order = "<" if sys.byteorder == "little" else ">"
    layout = np.dtype([("order", "u1"), ("type", f"{order}u4"), ("x", f"{order}f8"), ("y", f"{order}f8")])
    assert layout.itemsize == POINT_WKB_BYTES
    out = np.empty(len(coordinates), dtype=layout)
    out["order"] = 1 if order == "<" else 0
    out["type"] = 1
    out["x"] = coordinates[:, 0]
    out["y"] = coordinates[:, 1]
    return out.tobytes()


def _layout_holds() -> bool:
    """Whether shapely's points read and write as this module lays them
    out, checked once per process (per shapely module)."""
    global _CHECKED
    shapely = sys.modules.get("shapely")
    checked = _CHECKED
    if checked is not None and checked[0] is shapely:
        return checked[1]
    with _CHECK_LOCK:
        holds = _check_layout()
        _CHECKED = (shapely, holds)
    return holds


def _check_layout() -> bool:
    import numpy as np
    import shapely

    try:
        sample = np.array(
            [[1.5, -2.25], [-0.0, 0.0], [np.inf, -np.inf], [5e-324, -1e308], [123456.789, 0.1]], dtype=np.float64
        )
        points = points_from_coordinates(sample)
        back = _read_coordinates(points)
        expected = point_wkb(sample)
        read = shapely.from_wkb(shapely.to_wkb(points))
        return (
            back.tobytes() == sample.tobytes()
            and b"".join(shapely.to_wkb(points)) == expected
            and b"".join(shapely.to_wkb(points, hex=False, include_srid=True)) == expected
            and _read_coordinates(read).tobytes() == sample.tobytes()
            and (shapely.get_coordinate_dimension(points) == 2).all()
        )
    except Exception:  # noqa: BLE001 - any surprise: the WKB way
        return False
