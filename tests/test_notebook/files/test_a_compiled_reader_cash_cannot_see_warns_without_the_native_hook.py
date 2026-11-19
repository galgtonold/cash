"""A statement reading a file through a popular library's compiled reader, with
the native file hook off (as on macOS), warns and runs every time.

Such a reader (``cv2.imread``, ``netCDF4.Dataset``, DuckDB, GDAL through
pyogrio, fiona or osgeo) opens the file in compiled code, which only the
native file hook sees. Without it the statement was stored with no input and
a rerun after the file changed showed the old value, with no warning.
"""

from __future__ import annotations

import os
import warnings

import pytest

from cash.tracking import native_reads
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
shapely = pytest.importorskip("shapely")


@pytest.fixture
def hook_off():
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


def test_the_statement_warns_and_sees_an_edit(cash_magics, tmp_path, hook_off):
    path = tmp_path / "points.geojson"
    _write(path, 2)
    cells = [
        f"import time\nimport pyogrio\nP = {str(path)!r}",
        f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\nn = len(pyogrio.read_dataframe(P))",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[1], cells=cells)
    unseen = [str(w.message) for w in caught if getattr(w.message, "code", None) == "KEY-NATIVE-READ-UNSEEN"]
    assert unseen, [str(w.message) for w in caught]
    assert "pyogrio.read_dataframe" in unseen[0] and str(path) in unseen[0]
    assert "n = len(pyogrio.read_dataframe(P))" in unseen[0]
    assert cash_magics.shell.user_ns["n"] == 2

    _write(path, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, cells[1], cells=cells)
    assert cash_magics.shell.user_ns["n"] == 5
