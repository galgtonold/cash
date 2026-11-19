"""A cell reading a file through a popular library's compiled reader, in a
kernel without the native file hook (as on macOS), warns and reads the file
again after it changes.

The reader (here GDAL through pyogrio) opens the file in compiled code, which
only the native file hook sees: without it the cell was stored with no input,
and a rerun showed the old value with no warning.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.files]

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
shapely = pytest.importorskip("shapely")


def _write(path, n):
    frame = gpd.GeoDataFrame({"n": list(range(n))}, geometry=[shapely.Point(i, i) for i in range(n)])
    frame.to_file(path, driver="GeoJSON")


def test_the_cell_warns_and_shows_the_edit(nb_runner, monkeypatch):
    monkeypatch.setenv("CASH_NATIVE_FILE_HOOK", "0")
    _write(nb_runner.work_dir / "points.geojson", 2)
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import time\nimport pyogrio",
            f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\nn = len(pyogrio.read_dataframe('points.geojson'))\nprint('N', n)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "N 2" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
    assert "KEY-NATIVE-READ-UNSEEN" in nb_runner.get_raw_output(3), nb_runner.get_raw_output(3)

    _write(nb_runner.work_dir / "points.geojson", 5)
    nb_runner.run_cell(3)
    assert "N 5" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
