"""A callable of the user's handed to a library call changes the objects a
frame's cells hold when it changes its parameter in place, and cash must
then move the frame: a reader of ``df['o']`` below must not be served what
it read before.

What cash cannot read whole counts as changing: a call in the function
that cash cannot follow (a decorator's wrapper, a closure calling a closure,
a partial, a method, a dict-held function, a callable object, a module
attribute, ``self._set(d)``, ``exec``, a local alias, ``getattr``); a
lambda calling any of them; and any call that may run what it is handed
(``sorted(key=)``, ``starmap``, ``np.vectorize``, ``df.assign``,
``sort_values(key=)``, ``df.loc[f]``), its argument read through what it is
built from (``list(df['o'])``, ``pd.Series(df['o'])``). Each was served
stale with no warning. Callables cash reads whole and finds leaving their
parameter alone are still served.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = (
    "class O:\n    def __init__(self, i):\n        self.i = i\n"
    "    def __repr__(self):\n        return f'O{sorted(self.__dict__.items())}'\n"
    "df = pd.DataFrame({'x': np.arange(10.), 'k': [i % 2 for i in range(10)], "
    "'o': [{'i': i} for i in range(10)], 'p': [O(i) for i in range(10)]})\n"
    "Z = 0"
)
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
SETZ = "def setz(d):\n    if d['i'] == 3:\n        d['z'] = Z"
CHANGES = {
    "wraps_decorator": (
        "import functools\ndef deco(fn):\n    @functools.wraps(fn)\n    def w(*a, **k):\n        return fn(*a, **k)\n    return w\n@deco\n"
        + SETZ.replace("setz", "upd"),
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "plain_decorator": (
        "def deco(fn):\n    def w(d):\n        return fn(d)\n    return w\n@deco\n" + SETZ.replace("setz", "upd"),
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "closure_calls_closure": (
        "def make():\n    def inner(d):\n        if d['i'] == 3:\n            d['z'] = Z\n    def upd(d):\n        inner(d)\n    return upd\nupd = make()",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_calls_global_partial": (
        "import functools\ndef setk(d, k):\n    if d['i'] == 3:\n        d[k] = Z\nfix = functools.partial(setk, k='z')\ndef upd(d):\n    fix(d)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_calls_obj_method": (
        "class H:\n    def fix(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nh = H()\ndef upd(d):\n    h.fix(d)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_calls_dict_fn": (
        SETZ + "\nOPS = {'f': setz}\ndef upd(d):\n    OPS['f'](d)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_calls_callable_obj": (
        "class H:\n    def __call__(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nh = H()\ndef upd(d):\n    h(d)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "method_calls_self_helper": (
        "class U:\n    def upd(self, d):\n        self._set(d)\n    def _set(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nu = U()",
        "Z = {z}\n_ = df['o'].apply(u.upd)",
    ),
    "def_local_alias": (
        "def upd(d):\n    x = d\n    if x['i'] == 3:\n        x['z'] = Z",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_method_alias": (
        "def upd(d):\n    put = d.__setitem__\n    if d['i'] == 3:\n        put('z', Z)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_getattr_setitem": (
        "def upd(d):\n    if d['i'] == 3:\n        getattr(d, '__setitem__')('z', Z)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "def_ior": ("def upd(d):\n    if d['i'] == 3:\n        d |= {'z': Z}", "Z = {z}\n_ = df['o'].apply(upd)"),
    "def_exec": ("def upd(d):\n    if d['i'] == 3:\n        exec(\"d['z'] = Z\")", "Z = {z}\n_ = df['o'].apply(upd)"),
    "lambda_calls_partial": (
        "import functools\ndef setk(d, k):\n    if d['i'] == 3:\n        d[k] = Z\nfix = functools.partial(setk, k='z')",
        "Z = {z}\n_ = df['o'].map(lambda d: fix(d))",
    ),
    "lambda_calls_bound": (
        "class H:\n    def fix(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nh = H()",
        "Z = {z}\n_ = df['o'].map(lambda d: h.fix(d))",
    ),
    "lambda_calls_callable_obj": (
        "class H:\n    def __call__(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nh = H()",
        "Z = {z}\n_ = df['o'].map(lambda d: h(d))",
    ),
    "lambda_calls_dict_fn": (SETZ + "\nOPS = {'f': setz}", "Z = {z}\n_ = df['o'].map(lambda d: OPS['f'](d))"),
    "sorted_key": (
        "def setz(d):\n    if d['i'] == 3:\n        d['z'] = Z\n    return d['i']",
        "Z = {z}\n_ = sorted(df['o'], key=setz)",
    ),
    "max_key_partial": (
        "import functools\ndef setk(d, k):\n    if d['i'] == 3:\n        d[k] = Z\n    return d['i']\nfix = functools.partial(setk, k='z')",
        "Z = {z}\n_ = max(df['o'], key=fix)",
    ),
    "reduce_def": (
        "import functools\ndef acc(a, d):\n    if d['i'] == 3:\n        d['z'] = Z\n    return a",
        "Z = {z}\n_ = functools.reduce(acc, df['o'], 0)",
    ),
    "starmap": (
        "import itertools\ndef upd(d, k):\n    if d['i'] == 3:\n        d[k] = Z",
        "Z = {z}\n_ = list(itertools.starmap(upd, zip(df['o'], ['z'] * 10)))",
    ),
    "np_vectorize": (SETZ, "Z = {z}\n_ = np.vectorize(setz, otypes=[object])(df['o'].to_numpy())"),
    "np_frompyfunc": (SETZ, "Z = {z}\n_ = np.frompyfunc(setz, 1, 1)(df['o'])"),
    "unbound_series_apply": (SETZ, "Z = {z}\n_ = pd.Series.apply(df['o'], setz)"),
    "assign_callable_obj": (
        "class H:\n    def __call__(self, frame):\n        for d in frame['o']:\n            if d['i'] == 3:\n                d['z'] = Z\n        return 0\nh = H()",
        "Z = {z}\n_ = df.assign(y=h)",
    ),
    "assign_partial": (
        "import functools\ndef setall(frame, k):\n    for d in frame['o']:\n        if d['i'] == 3:\n            d[k] = Z\n    return 0\nfix = functools.partial(setall, k='z')",
        "Z = {z}\n_ = df.assign(y=fix)",
    ),
    "groupby_apply_obj": (
        "class H:\n    def __call__(self, g):\n        for d in g['o']:\n            if d['i'] == 3:\n                d['z'] = Z\n        return len(g)\nh = H()",
        "Z = {z}\n_ = df.groupby('k')[['o']].apply(h)",
    ),
    "sort_values_key_obj": (
        "class H:\n    def __call__(self, s):\n        for d in s:\n            if isinstance(d, dict) and d['i'] == 3:\n                d['z'] = Z\n        return s.map(str)\nh = H()",
        "Z = {z}\n_ = df.sort_values('o', key=h)",
    ),
    "loc_callable_partial": (
        "import functools\ndef pick(frame, k):\n    for d in frame['o']:\n        if d['i'] == 3:\n            d[k] = Z\n    return frame['x'] > 0\nfix = functools.partial(pick, k='z')",
        "Z = {z}\n_ = df.loc[fix]",
    ),
    "comprehension_partial": (
        "import functools\ndef setk(d, k):\n    if d['i'] == 3:\n        d[k] = Z\nfix = functools.partial(setk, k='z')",
        "Z = {z}\n_ = [fix(d) for d in df['o']]",
    ),
    "comprehension_bound": (
        "class H:\n    def fix(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nh = H()",
        "Z = {z}\n_ = [h.fix(d) for d in df['o']]",
    ),
    "for_loop_obj": (
        "class H:\n    def __call__(self, d):\n        if d['i'] == 3:\n            d['z'] = Z\nh = H()",
        "Z = {z}\nfor d in df['o']:\n    h(d)",
    ),
    "classmethod": (
        "class U:\n    @classmethod\n    def upd(cls, d):\n        if d['i'] == 3:\n            d['z'] = Z",
        "Z = {z}\n_ = df['o'].apply(U.upd)",
    ),
    "module_fn_via_attr": (
        "import types as _t\nM = _t.SimpleNamespace()\n" + SETZ + "\nM.setz = setz\ndef upd(d):\n    M.setz(d)",
        "Z = {z}\n_ = df['o'].apply(upd)",
    ),
    "map_list_of_column": (SETZ, "Z = {z}\n_ = list(map(setz, list(df['o'])))"),
    "map_tolist": (SETZ, "Z = {z}\n_ = list(map(setz, df['o'].tolist()))"),
    "series_ctor_apply": (SETZ, "Z = {z}\n_ = pd.Series(df['o']).apply(setz)"),
    "map_values": (SETZ, "Z = {z}\n_ = list(map(setz, df['o'].values))"),
    "np_array_apply": (SETZ, "Z = {z}\n_ = pd.Series(np.asarray(df['o'])).map(setz)"),
}
READERS = {
    "statement": [f"g = ({SLOW}, (df['o'].map(str) + '|' + df['p'].map(repr)).tolist())[1]"],
    "call": [
        f"def strs(d):\n    {SLOW}\n    return (d['o'].map(str) + '|' + d['p'].map(repr)).tolist()",
        "g = strs(df)",
    ],
}
#: Shapes read by a call too, not only by a statement.
CALL_READ = ["wraps_decorator", "def_exec", "lambda_calls_partial", "sorted_key", "series_ctor_apply"]


@pytest.mark.parametrize(
    ("change", "reader"),
    [(change, "statement") for change in CHANGES] + [(change, "call") for change in CALL_READ],
)
def test_the_reader_below_sees_the_change(cash_magics, change, reader):
    hold, edit = CHANGES[change]
    got = []
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
        want = (ns["df"]["o"].map(str) + "|" + ns["df"]["p"].map(repr)).tolist()
        got.append((z, ns["g"][3], want[3]))
    assert all(a == b for _, a, b in got), got


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
    "pure_partial_var": (
        "import functools\ndef times(d, k):\n    return d['i'] * k\npt = functools.partial(times, k=3)",
        "_ = df['o'].apply(pt)",
    ),
    "pure_bound": (
        "class U:\n    def __init__(self):\n        self.k = 2\n    def two(self, d):\n        return d['i'] * self.k\nu = U()",
        "_ = df['o'].apply(u.two)",
    ),
    "ufunc": ("", "_ = df['x'].apply(np.sqrt)"),
    "ufunc_obj": ("", "_ = df['o'].map(np.positive, na_action=None) if False else df['x'].map(np.square)"),
    "lambda_global": ("K = 5", "_ = df['o'].map(lambda d: d['i'] * K)"),
    "str_upper_map": ("S = pd.Series(['a', 'b'] * 5)", "_ = list(map(str.upper, S))"),
    "str_upper_frame_map": ("", "_ = list(map(str.upper, df['o'].map(str)))"),
    "pure_static": (
        "class U:\n    @staticmethod\n    def two(d):\n        return d['i'] * 2",
        "_ = df['o'].apply(U.two)",
    ),
    "map_def_tolist": ("def two(d):\n    return d['i'] * 2", "_ = list(map(two, df['o'].tolist()))"),
    "map_def_values": ("def two(d):\n    return d['i'] * 2", "_ = list(map(two, df['o'].values))"),
    "filter_def_list": ("def big(d):\n    return d['i'] > 2", "_ = list(filter(big, list(df['o'])))"),
    "map_len_strs": ("", "_ = list(map(len, df['o'].map(str)))"),
    "apply_def_head": ("def two(d):\n    return d['i'] * 2", "_ = df['o'].head(5).apply(two)"),
    "map_lambda_tolist": ("", "_ = list(map(lambda d: d['i'], df['o'].tolist()))"),
    "map_partial_tolist": (
        "import functools\ndef times(d, k):\n    return d['i'] * k",
        "_ = list(map(functools.partial(times, k=2), df['o'].tolist()))",
    ),
    "apply_obj_pure": (
        "class T:\n    def __call__(self, d):\n        return d['i'] * 2\nt = T()",
        "_ = df['o'].apply(t)",
    ),
}


@pytest.mark.parametrize("read", list(READS))
def test_a_callable_leaving_its_parameter_alone_is_still_served(cash_magics, read):
    hold, edit = READS[read]
    cells = ["import time\nimport numpy as np, pandas as pd", f"{FRAME}\n{hold}", edit, *READERS["statement"]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells[2:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]] == ["RESTORED"]
