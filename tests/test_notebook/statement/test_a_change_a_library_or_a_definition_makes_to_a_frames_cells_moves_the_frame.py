"""A helper that changes the objects of a frame's object column moves the
frame for a slow reader below, when the change is made through a library
(``rng.shuffle(v)``, ``np.add.at(v, ...)``, ``out=v``), or by code a class
body, a default argument or a decorator runs at definition time. A method
of a library's result was read as changing nothing it was handed, and the
statements a definition runs were skipped with the definition's body: the
reader was served the order or value the run before made."""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
FRAME = (
    "df = pd.DataFrame({'o': [{'i': i} for i in range(10)], "
    "'l': [[i, i + 1, i + 2, i + 3, i + 4] for i in range(10)], "
    "'v': [np.arange(5.) + i for i in range(10)]})"
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
    "control_a_plain_apply": (SETZ, "Z = {z}\n_ = df['o'].apply(setz)"),
}
READ = f"g = ({SLOW}, (df['o'].map(str) + df['l'].map(str) + df['v'].map(lambda a: str(list(a)))).tolist())[1]"


@pytest.mark.parametrize("case", list(CASES))
def test_the_reader_below_sees_the_change(cash_magics, case):
    helper, change = CASES[case]
    for z in (1, 2):
        cells = ["import time\nimport numpy as np, pandas as pd", f"{FRAME}\n{helper}", change.format(z=z), READ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        ns = cash_magics.shell.user_ns
        df = ns["df"]
        want = (df["o"].map(str) + df["l"].map(str) + df["v"].map(lambda a: str(list(a)))).tolist()
        assert ns["g"] == want, f"Z = {z}"
