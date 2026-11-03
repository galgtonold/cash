"""A statement that changes its input in place is stored with the state it
left the input in, and a hit writes that state back into the live object.

``W, H = fit_nmf(X, k)`` where ``fit_nmf`` writes into ``X`` ran on every
Run All (33 s a pass in the notebook that showed it): a hit would have handed
back ``W`` and ``H`` and left ``X`` as it was. The entry now keeps ``X``'s
state after the call, and a hit writes it into the object ``X`` is bound to,
so ``X`` keeps its identity and another name holding the same object sees
the change, as after a plain run.

Only objects a write in place fully restores are stored this way: NumPy
arrays, pandas frames and Series whose index, columns and dtypes the
statement left alone, and lists, dicts and sets of plain values. Anything
else still runs every time.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

#: ``fill(v)`` changes *v* in place and returns a summary; each run appends
#: one byte to ``LOG``, which a hit cannot skip or replay.
HELPERS = (
    "import os, time\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "def _ran():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "def fill(v):\n"
    "    _ran()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    if isinstance(v, np.ndarray) and v.dtype == object:\n"
    "        v[2] = 'filled'\n"
    "        return 1\n"
    "    if isinstance(v, np.ndarray):\n"
    "        np.maximum(v, 0.5, out=v)\n"
    "        return float(v.sum())\n"
    "    if isinstance(v, pd.DataFrame):\n"
    "        v['a'] *= 2\n"
    "        v.loc[0, 'b'] = 'z'\n"
    "        return float(v['a'].sum())\n"
    "    if isinstance(v, pd.Series):\n"
    "        v.iloc[1] = 99.0 if v.dtype == float else 'z'\n"
    "        return len(v)\n"
    "    if isinstance(v, list):\n"
    "        v.sort(reverse=True)\n"
    "        v.append(-1)\n"
    "        return len(v)\n"
    "    if isinstance(v, dict):\n"
    "        v['n'] = v.get('n', 0) + 7\n"
    "        del v['gone']\n"
    "        return len(v)\n"
    "    if isinstance(v, set):\n"
    "        v.add('new')\n"
    "        v.discard('a')\n"
    "        return len(v)\n"
    "    v.n = getattr(v, 'n', 0) + 7\n"
    "    return 1\n"
    "class Box:\n"
    "    pass\n"
)

#: How each supported kind is made, and how its value is compared.
KINDS = {
    "array": "v = np.linspace(0, 1, 11)",
    "frame": "v = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': ['x', 'y', 'w']})",
    "series": "v = pd.Series([1.0, 2.0, 3.0], index=['p', 'q', 'r'], name='s')",
    "list": "v = [3, 1, 2]",
    "dict": "v = {'n': 1, 'gone': 2, 'k': 'v'}",
    "set": "v = {'a', 'b', 'c'}",
}

CALL = "r = fill(v)"
READ = "seen = repr(alias) + repr(r)"


def _plain(cells: list[str], log: str) -> dict:
    """What a restart and Run All in plain Python leaves."""
    ns: dict = {"LOG": log}
    for cell in cells:
        exec(cell, ns)
    return ns


def _runs(log) -> int:
    return len(log.read_bytes()) if log.exists() else 0


def _run_all(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def _same(a, b) -> bool:
    if hasattr(a, "equals"):
        return (
            type(a) is type(b)
            and a.equals(b)
            and list(a.dtypes if a.ndim == 2 else [a.dtype]) == list(b.dtypes if b.ndim == 2 else [b.dtype])
        )
    if hasattr(a, "tolist"):
        return type(a) is type(b) and a.dtype == b.dtype and a.tolist() == b.tolist()
    return type(a) is type(b) and a == b


@pytest.fixture
def notebook(cash_magics, tmp_path):
    cash_magics.cash_on("")
    log = tmp_path / "runs.log"
    cash_magics.shell.user_ns["LOG"] = str(log)
    return cash_magics, log


@pytest.mark.parametrize("kind", list(KINDS))
def test_the_change_is_stored_and_written_back_into_the_live_object(notebook, tmp_path, kind):
    magics, log = notebook
    ns = magics.shell.user_ns
    cells = [HELPERS, KINDS[kind] + "\nalias = v", CALL, READ]
    _run_all(magics, cells)
    assert _runs(log) == 1
    # The second Run All: the cell above binds a new object in the state
    # before the call; the hit writes the stored state into that object.
    for cell in cells[:2]:
        run_cash_cell(magics, cell, cells=cells)
    # By id: a reference held here would count as another holder.
    live = id(ns["v"])
    for cell in cells[2:]:
        run_cash_cell(magics, cell, cells=cells)
    assert _runs(log) == 1, f"the {kind} the call changes in place sent it back to run"
    assert id(ns["v"]) == live, "the hit rebound the name to a copy"
    assert ns["alias"] is ns["v"], "another name holding the same object lost it"
    plain = _plain(cells, str(tmp_path / "plain.log"))
    assert _same(ns["v"], plain["v"]), (ns["v"], plain["v"])
    assert ns["r"] == plain["r"]
    assert ns["seen"] == plain["seen"]


def test_the_p13_shape_a_function_writing_into_its_array(notebook):
    """``W, H = fit_nmf(X, k)``: two results, and the array it writes into."""
    magics, log = notebook
    ns = magics.shell.user_ns
    fit = (
        "def fit_nmf(X, k):\n"
        "    _ran()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    np.clip(X, 0.25, None, out=X)\n"
        "    return X[:k].copy(), X.sum() * k\n"
    )
    cells = [HELPERS + fit, "X = np.arange(12, dtype=float).reshape(3, 4) / 20\nX0 = X", "W, H = fit_nmf(X, 2)"]
    for _ in range(2):
        _run_all(magics, cells)
    assert _runs(log) == 1
    assert ns["X0"] is ns["X"]
    assert ns["X"].min() == 0.25 and ns["W"].tolist() == ns["X"][:2].tolist()
    assert ns["H"] == pytest.approx(float(ns["X"].sum()) * 2)


def test_a_changed_input_misses(notebook):
    """The entry is keyed on the input before the call: other values run it."""
    magics, log = notebook
    ns = magics.shell.user_ns
    cells = [HELPERS, "v = np.linspace(0, 1, 11)", CALL]
    _run_all(magics, cells)
    cells[1] = "v = np.linspace(0, 2, 11)"
    _run_all(magics, cells)
    assert _runs(log) == 2
    assert ns["v"].tolist() == np.maximum(np.linspace(0, 2, 11), 0.5).tolist()


def test_an_input_changed_in_place_before_the_call_misses(notebook):
    magics, log = notebook
    cells = [HELPERS, "v = np.linspace(0, 1, 11)", CALL]
    _run_all(magics, cells)
    cells.insert(2, "v[3] = 7.0")
    _run_all(magics, cells)
    assert _runs(log) == 2
    assert magics.shell.user_ns["r"] == pytest.approx(float(np.maximum(np.linspace(0, 1, 11), 0.5).sum()) + 6.5)


def test_a_view_of_an_array_is_written_through_to_its_base(notebook, tmp_path):
    magics, log = notebook
    ns = magics.shell.user_ns
    cells = [HELPERS, "base = np.linspace(0, 1, 11)\nv = base[2:8]", CALL, "total = float(base.sum())"]
    for _ in range(2):
        _run_all(magics, cells)
    plain = _plain(cells, str(tmp_path / "plain.log"))
    assert np.shares_memory(ns["v"], ns["base"]), "the view was rebound off its base"
    assert ns["base"].tolist() == plain["base"].tolist()
    assert ns["total"] == plain["total"]
    assert ns["r"] == plain["r"]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param("v = Box()", id="a class instance"),
        pytest.param("v = np.array([1, 'a', None], dtype=object)", id="an array of objects"),
        pytest.param("v = [[3], [1], [2]]", id="a list of lists"),
        pytest.param("v = {'n': 1, 'gone': [2], 'k': 'v'}", id="a dict holding a list"),
        pytest.param("v = pd.Series(['a', 'b', 'c'])", id="a Series of strings"),
        pytest.param("v = np.linspace(0, 1, 11)\nrows = [v]", id="an array a list holds"),
    ],
)
def test_what_a_write_in_place_cannot_restore_still_runs_every_time(notebook, tmp_path, make):
    magics, log = notebook
    ns = magics.shell.user_ns
    cells = [HELPERS, make, CALL]
    for _ in range(2):
        _run_all(magics, cells)
    assert _runs(log) == 2
    plain = _plain(cells, str(tmp_path / "plain.log"))
    assert repr(ns["r"]) == repr(plain["r"])


def test_a_frame_given_a_new_column_runs_every_time(notebook, tmp_path):
    """A column added cannot be put into the frame in place: not stored."""
    magics, log = notebook
    ns = magics.shell.user_ns
    add = (
        "def add(df):\n    _ran()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    df['c'] = df['a'] * 3\n    return 1\n"
    )
    cells = [HELPERS + add, "df = pd.DataFrame({'a': [1.0, 2.0]})", "r = add(df)"]
    for _ in range(2):
        _run_all(magics, cells)
    assert _runs(log) == 2
    assert list(ns["df"].columns) == ["a", "c"]


def test_the_upstream_rebuild_writes_into_the_live_object_too(notebook):
    """A cell below run alone, with a result of the call gone: the upstream
    check restores the statement from its entry, and writes the stored state
    into ``v`` as a hit does, rather than binding ``v`` to a copy."""
    magics, log = notebook
    ns = magics.shell.user_ns
    read = "seen = repr(v) + repr(r)"
    cells = [HELPERS, KINDS["array"], CALL, read]
    _run_all(magics, cells)
    del ns["r"]
    live = id(ns["v"])
    run_cash_cell(magics, read, cells=cells)
    assert _runs(log) == 1
    assert id(ns["v"]) == live, "the rebuild bound v to a copy"
    assert ns["v"].min() == 0.5 and ns["r"] == float(ns["v"].sum())


def test_a_refused_store_still_moves_a_container_holding_the_input(notebook):
    """``config = {'features': features}`` then ``features.append(x)``: the
    dict holding the list refuses the store, and the statement runs every
    time as before, its receiver moving ``config`` on, so a cell reading
    ``config`` misses when the change is edited."""
    magics, _ = notebook
    ns = magics.shell.user_ns
    slow = f"import time\ndef slow(v):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return repr(v)"
    cells = [slow, "features = ['age']\nconfig = {'features': features}\nfeatures.append('zip')", "r = slow(config)"]
    _run_all(magics, cells)
    assert ns["r"] == "{'features': ['age', 'zip']}"
    cells[1] = cells[1].replace("zip", "city")
    _run_all(magics, cells)
    assert ns["r"] == "{'features': ['age', 'city']}"
