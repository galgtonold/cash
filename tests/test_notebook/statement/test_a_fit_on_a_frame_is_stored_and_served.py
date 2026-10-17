"""A model fitted on a frame is stored and served, under pandas 2 as under 3.

Under pandas 2 a fitted model's ``feature_names_in_`` is the frame's own
column array (``X.columns._data``), so every fit on a frame was refused as
sharing an object with ``X`` and ran again each time. An index is
immutable, so holding its data is holding a value (K10): the fit is now
stored. A changed frame still misses, and changing the frame after the fit
leaves the model as it was fitted.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

#: A cost above what the disk tier keeps, so a restart restores the model.
SLOW = f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 1)[1]"
DATA = (
    "import time\nimport numpy as np\nimport pandas as pd\n"
    "from sklearn.linear_model import LinearRegression\n"
    "df = pd.DataFrame({'a': np.arange(30.0), 'b': np.arange(30.0) % 7})\n"
    "y = 2 * df['a'] + 3 * df['b']"
)
FIT = "model = LinearRegression(fit_intercept=" + SLOW + " > 0).fit(df, y)"
NAMES = ["time", "np", "pd", "LinearRegression", "df", "y", "model"]


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics, cells: list[str]) -> list[str | None]:
    """Run *cells*; the status of each one's last statement."""
    out = []
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def _fitted(magics) -> tuple[list[str], list[float]]:
    model = magics.shell.user_ns["model"]
    return list(model.feature_names_in_), model.coef_.round(6).tolist()


def test_a_fit_on_a_frame_is_served_in_the_session_and_after_a_restart(cash_magics, clean_backend):
    assert _run(cash_magics, [DATA, FIT])[1] == CacheStatus.COMPUTED
    assert _run(cash_magics, [DATA, FIT])[1] == CacheStatus.RESTORED
    _restart(cash_magics)

    assert _run(cash_magics, [DATA, FIT])[1] == CacheStatus.RESTORED
    assert _fitted(cash_magics) == (["a", "b"], [2.0, 3.0])


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(DATA.replace("% 7", "% 5"), id="values"),
        pytest.param(DATA.replace("'b': ", "'c': ").replace("df['b']", "df['c']"), id="columns"),
        pytest.param(DATA + "\ndf = df.rename(columns={'b': 'c'})", id="rename"),
    ],
)
def test_a_changed_frame_misses(cash_magics, clean_backend, change):
    _run(cash_magics, [DATA, FIT])
    _restart(cash_magics)

    assert _run(cash_magics, [change, FIT])[1] != CacheStatus.RESTORED
    names, coef = _fitted(cash_magics)
    assert names == list(cash_magics.shell.user_ns["df"].columns)
    assert coef == [2.0, 3.0], "fitted on the changed frame"


def test_changing_the_frame_after_the_fit_leaves_the_served_model(cash_magics, clean_backend):
    _run(cash_magics, [DATA, FIT])
    run_cash_cell(cash_magics, "df.rename(columns={'a': 'z'}, inplace=True)\ndf['b'] *= 0")
    assert _fitted(cash_magics) == (["a", "b"], [2.0, 3.0])
    _restart(cash_magics)

    assert _run(cash_magics, [DATA, FIT])[1] == CacheStatus.RESTORED
    assert _fitted(cash_magics) == (["a", "b"], [2.0, 3.0])
