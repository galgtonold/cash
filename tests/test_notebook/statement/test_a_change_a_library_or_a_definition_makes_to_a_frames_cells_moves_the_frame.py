"""A helper that changes the objects of a frame's object column moves the
frame for a slow reader below, when the change is made through a library
(``rng.shuffle(v)``, ``np.add.at(v, ...)``, ``out=v``), or by code a class
body, a default argument or a decorator runs at definition time. A method
of a library's result was read as changing nothing it was handed, and the
statements a definition runs were skipped with the definition's body: the
reader was served the order or value the run before made.

What a library's call does to what it is handed is not read from its name:
a positional ``out`` slot, ``copy=False``, an unbound method
(``dict.update(d, ...)``) or a view of the parameter (``np.asarray(v)``)
changes it as ``out=`` does. A call handed a parameter is watched around
the run instead, so a pure one (``queue.Queue().put(v)``) is still
served."""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
FRAME = (
    "df = pd.DataFrame({'o': [{'i': i} for i in range(10)], "
    "'l': [[i, i + 1, i + 2, i + 3, i + 4] for i in range(10)], "
    "'v': [np.arange(5.) + i for i in range(10)], "
    "'n': [np.array([np.nan, i]) for i in range(10)]})"
)
SETZ = "def setz(d):\n    if d['i'] == 3:\n        d['z'] = Z"
CASES = {
    "a_class_body": (SETZ, "Z = {z}\nclass K:\n    r = df['o'].apply(setz)"),
    "a_default_argument": (SETZ, "Z = {z}\ndef k(r=df['o'].apply(setz)):\n    return r"),
    "a_decorator": (
        SETZ + "\ndef deco(r):\n    return lambda f: f",
        "Z = {z}\n@deco(df['o'].apply(setz))\ndef k():\n    return 1",
    ),
    "a_shuffle_of_a_new_random": (
        "import random\ndef upd(l):\n    random.Random(Z).shuffle(l)",
        "Z = {z}\n_ = df['l'].apply(upd)",
    ),
    "a_shuffle_of_a_local_generator": (
        "def upd(v):\n    rng = np.random.default_rng(Z)\n    rng.shuffle(v)",
        "Z = {z}\n_ = df['v'].apply(upd)",
    ),
    "a_shuffle_of_a_new_generator": (
        "def upd(v):\n    np.random.default_rng(Z).shuffle(v)",
        "Z = {z}\n_ = df['v'].apply(upd)",
    ),
    "an_add_at": ("def upd(v):\n    np.add.at(v, [0], Z)", "Z = {z}\n_ = df['v'].apply(upd)"),
    "an_out_argument": ("def upd(v):\n    np.multiply(v, Z, out=v)", "Z = {z}\n_ = df['v'].apply(upd)"),
    "an_unbound_dict_update": ("def upd(d):\n    dict.update(d, z=Z)", "Z = {z}\n_ = df['o'].apply(upd)"),
    "an_unbound_list_append": ("def upd(l):\n    list.append(l, Z)", "Z = {z}\n_ = df['l'].apply(upd)"),
    "an_unbound_list_insert": ("def upd(l):\n    list.insert(l, 0, Z)", "Z = {z}\n_ = df['l'].apply(upd)"),
    "an_unbound_array_fill": ("def upd(v):\n    np.ndarray.fill(v, Z)", "Z = {z}\n_ = df['v'].apply(upd)"),
    "a_positional_out": ("def upd(v):\n    np.add(v, Z, v)", "Z = {z}\n_ = df['v'].apply(upd)"),
    "a_positional_clip_out": ("def upd(v):\n    np.clip(v, -100, Z, v)", "Z = {z}\n_ = df['v'].apply(upd)"),
    "no_copy": ("def upd(n):\n    np.nan_to_num(n, copy=False, nan=Z)", "Z = {z}\n_ = df['n'].apply(upd)"),
    "a_fill_of_a_view": ("def upd(v):\n    np.asarray(v).fill(Z)", "Z = {z}\n_ = df['v'].apply(upd)"),
    "a_series_over_it_clipped_in_place": (
        "def upd(v):\n    pd.Series(v, copy=False).clip(upper=Z, inplace=True)",
        "Z = {z}\n_ = df['v'].apply(upd)",
    ),
    "a_write_into_a_buffer_view": (
        "def upd(v):\n    np.frombuffer(v, dtype=float)[0:1] = Z",
        "Z = {z}\n_ = df['v'].apply(upd)",
    ),
    "control_a_plain_apply": (SETZ, "Z = {z}\n_ = df['o'].apply(setz)"),
}
SHOWN = (
    "(df['o'].map(str) + df['l'].map(str) + df['v'].map(lambda a: str(list(a))) + df['n'].map(lambda a: str(list(a))))"
)
READ = f"g = ({SLOW}, {SHOWN}.tolist())[1]"


@pytest.mark.parametrize("case", list(CASES))
def test_the_reader_below_sees_the_change(cash_magics, case):
    helper, change = CASES[case]
    for z in (1, 2):
        cells = ["import time\nimport numpy as np, pandas as pd", f"{FRAME}\n{helper}", change.format(z=z), READ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        ns = cash_magics.shell.user_ns
        want = eval(f"{SHOWN}.tolist()", {"df": ns["df"]})
        assert ns["g"] == want, f"Z = {z}"


def test_a_library_call_that_leaves_its_argument_alone_is_served(cash_magics):
    """``queue.Queue().put(v)`` leaves ``v`` as it was: watched, not refused
    for its name."""
    cells = [
        "import time, queue\nimport numpy as np, pandas as pd",
        f"{FRAME}\ndef upd(v):\n    q = queue.Queue()\n    q.put(v)\n    return q.qsize()",
        f"_ = ({SLOW}, df['v'].apply(upd))[1]",
    ]
    statuses = []
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        statuses.append([str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]])
    assert statuses[-1] == ["RESTORED"], statuses
