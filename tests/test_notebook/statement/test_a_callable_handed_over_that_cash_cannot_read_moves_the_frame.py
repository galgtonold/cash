"""A callable handed to a method that runs it over a frame's object cells
(``df['o'].apply(f)``, ``map(f, df['o'])``, ``df.pipe(f)``) may change
those cells in place. One cash reads (a function of the user's, a lambda)
moves the frame when it changes its parameter. One it cannot read as
leaving its parameter as it was -- a partial, a bound or static method, a
callable object, a function taken out of a dict or list -- counts as
changing it: a reader of the frame then reruns, never serves the cells as
they were before the change."""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = (
    "class O:\n    def __init__(self, i):\n        self.i = i\n"
    "    def __repr__(self):\n        return f'O{sorted(self.__dict__.items())}'\n"
    "df = pd.DataFrame({'x': np.arange(10.), 'o': [{'i': i} for i in range(10)], 'p': [O(i) for i in range(10)]})"
)
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
UPD = "def upd(d):\n    if d['i'] == 3:\n        d['z'] = Z"
UPD_Z = "import functools\ndef upd(d, z):\n    if d['i'] == 3:\n        d['z'] = z"
CHANGES = {
    "partial_inline": (UPD_Z, "_ = df['o'].apply(functools.partial(upd, z={z}))"),
    "partial_variable": (UPD_Z, "p = functools.partial(upd, z={z})\n_ = df['o'].apply(p)"),
    "partial_given_keywords": (UPD_Z + "\np = functools.partial(upd, z=0)", "Zp = {z}\n_ = df['o'].apply(p, z=Zp)"),
    "bound_method": (
        "class U:\n    def upd(self, d):\n        if d['i'] == 3:\n            d['z'] = self.z\nu = U()",
        "u.z = {z}\n_ = df['o'].apply(u.upd)",
    ),
    "staticmethod": (
        "class U:\n    @staticmethod\n    def upd(d):\n        if d['i'] == 3:\n            d['z'] = Z",
        "Z = {z}\n_ = df['o'].apply(U.upd)",
    ),
    "callable_object": (
        "class U:\n    def __call__(self, d):\n        if d['i'] == 3:\n            d['z'] = self.z\nu = U()",
        "u.z = {z}\n_ = df['o'].apply(u)",
    ),
    "def_in_dict": (UPD + "\nF = {'u': upd}", "Z = {z}\n_ = df['o'].apply(F['u'])"),
    "def_in_list": (UPD + "\nFS = [upd]", "Z = {z}\n_ = df['o'].apply(FS[0])"),
    "builtin_map": (UPD, "Z = {z}\n_ = list(map(upd, df['o']))"),
    "builtin_filter": (UPD, "Z = {z}\n_ = list(filter(upd, df['o']))"),
    "lambda_dunder_setitem": ("", "_ = df['o'].map(lambda d: d.__setitem__('z', {z}) if d['i'] == 3 else None)"),
    "lambda_vars_update": ("", "_ = df['p'].map(lambda o: vars(o).update(z={z}) if o.i == 3 else None)"),
    "def_setattr": ("def upd(o):\n    if o.i == 3:\n        setattr(o, 'z', Z)", "Z = {z}\n_ = df['p'].apply(upd)"),
    "pipe_loop": (
        "def f(frame):\n    for d in frame['o']:\n        if d['i'] == 3:\n            d['z'] = Z\n    return frame",
        "Z = {z}\n_ = df.pipe(f)",
    ),
}
READERS = {
    "statement": [f"g = ({SLOW}, (df['o'].map(str) + '|' + df['p'].map(repr)).tolist())[1]"],
    "call": [
        f"def strs(d):\n    {SLOW}\n    return (d['o'].map(str) + '|' + d['p'].map(repr)).tolist()",
        "g = strs(df)",
    ],
}


@pytest.mark.parametrize("reader", list(READERS))
@pytest.mark.parametrize("change", list(CHANGES))
def test_a_reader_after_the_change_sees_it(cash_magics, change, reader):
    hold, edit = CHANGES[change]
    for z in (1, 2):
        cells = [
            "import time\nimport numpy as np, pandas as pd",
            f"{FRAME}\n{hold}",
            edit.format(z=z),
            *READERS[reader],
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        ns = cash_magics.shell.user_ns
        assert ns["g"] == (ns["df"]["o"].map(str) + "|" + ns["df"]["p"].map(repr)).tolist(), (z, ns["g"][3])


READS = {
    "builtin": ("", "_ = df['o'].map(str)"),
    "library": ("", "_ = df['x'].apply(np.log1p)"),
    "lambda": ("", "_ = df['o'].map(lambda d: d['i'] * 2)"),
    "def": ("def two(d):\n    return d['i'] * 2", "_ = df['o'].apply(two)"),
    "partial_of_def": (
        "import functools\ndef times(d, k):\n    return d['i'] * k",
        "_ = df['o'].apply(functools.partial(times, k=2))",
    ),
    "def_in_dict": ("def two(d):\n    return d['i'] * 2\nF = {'t': two}", "_ = df['o'].apply(F['t'])"),
    "builtin_map": ("def two(d):\n    return d['i'] * 2", "_ = list(map(two, df['o']))"),
    "pipe": ("def size(frame):\n    return len(frame)", "_ = df.pipe(size)"),
}


@pytest.mark.parametrize("read", list(READS))
def test_a_callable_read_as_leaving_its_parameter_keeps_the_reader_served(cash_magics, read):
    hold, edit = READS[read]
    cells = ["import time\nimport numpy as np, pandas as pd", f"{FRAME}\n{hold}", edit, *READERS["statement"]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells[2:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]] == ["RESTORED"]


def test_a_def_that_hands_a_changing_callable_changes_nothing_when_defined():
    """A ``def``'s body runs when it is called: defining one that hands a
    callable changing its parameter changes no variable, not even one that
    shares the parameter's name (rerunning the ``def`` would move it)."""
    import ast

    import numpy as np
    import pandas as pd

    from cash.analysis.mutation_effects import classify_receivers

    def bump(col):
        col += 1.0

    source = "def prep(df):\n    df.apply(bump, raw=True)\n    return len(df)"
    namespace = {"bump": bump, "df": pd.DataFrame({"a": np.arange(3.0)})}
    found = classify_receivers(ast.parse(source), namespace, lambda: None, arguments=())
    assert "df" not in found.mutated
