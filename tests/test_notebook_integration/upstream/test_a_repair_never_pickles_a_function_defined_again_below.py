"""A repair never pickles a value holding a function a cell below defined again.

Hands-On ML's chapter 2 dumps its fitted pipeline (``joblib.dump(final_model,
...)``) and then, in the cell that loads it back, defines ``column_ratio``
again. A plain run pickles the pipeline above that cell, where the name holds
the function the pipeline holds. A repair re-runs the dump below it: pickle
finds the function by its name, which holds the new one, and refuses ("it's
not the same object as __main__.column_ratio"). The cell below failed with
``UpstreamStateError`` on a notebook that runs top to bottom, and the dump
left a half-written file behind.

Cash cannot re-run such a statement as a run from the top does, so it re-runs
nothing and the cell runs as written, with ``NOTEBOOK-REPAIR-GAP``: before
anything runs for a file writer, after the pickle refused for any other
statement.
"""

import pickle
import sys
import types

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

ON = "import cash\n%cash_on\n%cash_badge print"
SETUP = "import time, pickle\nimport numpy as np\nX = np.random.RandomState(0).rand(400, 3) + 1.0\ny = X[:, 0] * 2"
DEFS = (
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.preprocessing import FunctionTransformer, StandardScaler\n"
    "from sklearn.linear_model import Ridge\n"
    "def column_ratio(X):\n    return X[:, [0]] / X[:, [1]]\n\n"
    "def ratio_pipeline():\n    return make_pipeline(FunctionTransformer(column_ratio), StandardScaler())"
)
MODEL = "model = make_pipeline(ratio_pipeline(), Ridge())"
FIT = f"model.fit((time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2}), X)[1], y)"
FINAL = "final_model = model"
DUMP = "import joblib\njoblib.dump(final_model, 'model.pkl'{extra})"
BLOB = "blob = pickle.dumps(final_model{extra})"
RELOAD = (
    "import joblib\n"
    "def column_ratio(X):\n    return X[:, [0]] / X[:, [1]]\n\n"
    "reloaded = joblib.load('model.pkl')\n"
    "predictions = reloaded.predict(X[:5])"
)
SHOW = "print('P', round(float(predictions.sum()), 6), type(pickle.loads(blob)).__name__)"


def _plain(cells: list[str], tmp_path, monkeypatch) -> str:
    """What the last cell prints when the notebook runs from the top: in a
    module of its own, which pickle finds its functions in by name, and in
    a folder of its own: *tmp_path* is the notebook's."""
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    monkeypatch.chdir(plain_dir)
    module = types.ModuleType("plain_notebook")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for code in cells[1:-1]:
        exec(code, module.__dict__)
    ns = module.__dict__
    return f"P {round(float(ns['predictions'].sum()), 6)} {type(pickle.loads(ns['blob'])).__name__}"


def _cells(dump: str = "", blob: str = "") -> list[str]:
    return [
        ON,
        SETUP,
        DEFS,
        MODEL,
        FIT,
        FINAL,
        DUMP.format(extra=dump),
        BLOB.format(extra=blob),
        RELOAD,
        SHOW,
    ]


def _run(nb_runner, restart: bool) -> None:
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()
    if restart:
        nb_runner.settle_writes()
        nb_runner.restart()
        nb_runner._init_cash()
        nb_runner.run_all()


@pytest.mark.parametrize("restart", [False, True], ids=["same_kernel", "after_a_restart"])
def test_a_dump_edited_above_runs_the_cell_below_as_written(nb_runner, tmp_path, monkeypatch, restart):
    """The dump's cell is edited and only the last cell runs: the repair
    would re-run the dump, after ``column_ratio`` was defined again."""
    _run(nb_runner, restart)
    dumped = (nb_runner.work_dir / "model.pkl").read_bytes()
    nb_runner.set_cell_source(7, DUMP.format(extra=", compress=3"))
    nb_runner.run_cells([10])

    raw = nb_runner.get_raw_output(10)
    assert "UpstreamStateError" not in raw, raw[-1500:]
    assert _plain(_cells(dump=", compress=3"), tmp_path, monkeypatch) in nb_runner.get_output(10), raw[-1500:]
    assert "NOTEBOOK-REPAIR-GAP" in raw, raw[-1500:]
    # Nothing re-ran: the dump did not start a file it could not finish.
    assert (nb_runner.work_dir / "model.pkl").read_bytes() == dumped


@pytest.mark.parametrize("restart", [False, True], ids=["same_kernel", "after_a_restart"])
def test_a_pickle_kept_in_a_variable_runs_the_cell_below_as_written(nb_runner, tmp_path, monkeypatch, restart):
    """``pickle.dumps`` writes no file, so the repair runs it and it fails
    there: the cell still runs as written, not into ``UpstreamStateError``."""
    _run(nb_runner, restart)
    nb_runner.set_cell_source(8, BLOB.format(extra=", protocol=4"))
    nb_runner.run_cells([10])

    raw = nb_runner.get_raw_output(10)
    assert "UpstreamStateError" not in raw, raw[-1500:]
    assert _plain(_cells(blob=", protocol=4"), tmp_path, monkeypatch) in nb_runner.get_output(10), raw[-1500:]
    assert "NOTEBOOK-REPAIR-GAP" in raw, raw[-1500:]
